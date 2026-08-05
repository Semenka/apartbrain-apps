from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

LOG = logging.getLogger("apartbrain-speaker-gate")
SAMPLE_RATE = 16000


class SpeakerGate:
    """Local VAD, opt-in speaker enrollment, and speaker verification."""

    def __init__(
        self,
        speaker_dir: Path,
        embedding_model: Path,
        vad_model: Path,
        allowed_speakers: list[str],
        threshold: float,
        enrollment_seconds: int = 10,
    ) -> None:
        self.speaker_dir = speaker_dir
        self.allowed_speakers = allowed_speakers
        self.threshold = threshold
        self.enrollment_samples_required = enrollment_seconds * SAMPLE_RATE
        self.lock = threading.RLock()
        self.enrollment_target: str | None = None
        self.enrollment_samples: list[np.ndarray] = []

        extractor_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model),
            num_threads=2,
            debug=False,
            provider="cpu",
        )
        if not extractor_config.validate():
            raise RuntimeError(f"Invalid speaker embedding model: {embedding_model}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(extractor_config)

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(vad_model)
        vad_config.silero_vad.min_silence_duration = 0.35
        vad_config.silero_vad.min_speech_duration = 0.5
        vad_config.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(
            vad_config,
            buffer_size_in_seconds=120,
        )
        self.vad_window_size = vad_config.silero_vad.window_size
        self.vad_buffer = np.empty(0, dtype=np.float32)
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.enrolled: set[str] = set()
        self.reload_enrollments()

    def reload_enrollments(self) -> None:
        with self.lock:
            self.speaker_dir.mkdir(parents=True, exist_ok=True)
            manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
            enrolled: set[str] = set()
            for name in self.allowed_speakers:
                samples = []
                for path in sorted(self.speaker_dir.glob(f"{name}-*.wav")):
                    audio = self._read_wav(path)
                    if len(audio) >= SAMPLE_RATE:
                        samples.append(self._embedding(audio))
                if samples:
                    embedding = np.mean(samples, axis=0)
                    if manager.add(name, embedding):
                        enrolled.add(name)
            self.manager = manager
            self.enrolled = enrolled
            LOG.info("Enrolled speakers: %s", ", ".join(sorted(enrolled)) or "none")

    def start_enrollment(self, name: str) -> None:
        if name not in self.allowed_speakers:
            raise ValueError(f"Unknown speaker: {name}")
        with self.lock:
            self.enrollment_target = name
            self.enrollment_samples = []

    def cancel_enrollment(self) -> None:
        with self.lock:
            self.enrollment_target = None
            self.enrollment_samples = []

    def status(self) -> dict:
        with self.lock:
            collected = sum(len(samples) for samples in self.enrollment_samples)
            return {
                "enrolled_speakers": sorted(self.enrolled),
                "enrollment_target": self.enrollment_target,
                "enrollment_progress": min(
                    1.0,
                    collected / max(1, self.enrollment_samples_required),
                ),
            }

    def accept_monitor_audio(self, samples: np.ndarray) -> list[str]:
        """Accept 16 kHz mono float32 audio and return verified speaker names."""
        with self.lock:
            self.vad_buffer = np.concatenate((self.vad_buffer, samples))
            while len(self.vad_buffer) >= self.vad_window_size:
                self.vad.accept_waveform(self.vad_buffer[: self.vad_window_size])
                self.vad_buffer = self.vad_buffer[self.vad_window_size :]

            matches: list[str] = []
            while not self.vad.empty():
                speech = np.asarray(self.vad.front.samples, dtype=np.float32)
                self.vad.pop()
                if len(speech) < SAMPLE_RATE:
                    continue
                if self.enrollment_target:
                    self.enrollment_samples.append(speech)
                    if (
                        sum(len(item) for item in self.enrollment_samples)
                        >= self.enrollment_samples_required
                    ):
                        self._finish_enrollment()
                    continue
                name = self.identify_samples(speech)
                if name:
                    matches.append(name)
            return matches

    def identify_samples(self, samples: np.ndarray) -> str | None:
        if len(samples) < SAMPLE_RATE // 2:
            return None
        with self.lock:
            if not self.enrolled:
                return None
            embedding = self._embedding(samples)
            name = self.manager.search(embedding, threshold=self.threshold)
            return name or None

    def _finish_enrollment(self) -> None:
        name = self.enrollment_target
        if not name:
            return
        audio = np.concatenate(self.enrollment_samples)
        stamp = len(list(self.speaker_dir.glob(f"{name}-*.wav"))) + 1
        path = self.speaker_dir / f"{name}-{stamp}.wav"
        self._write_wav(path, audio)
        self.enrollment_target = None
        self.enrollment_samples = []
        self.reload_enrollments()
        LOG.info("Completed local voice enrollment for %s", name)

    def _embedding(self, samples: np.ndarray) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(
            sample_rate=SAMPLE_RATE,
            waveform=np.ascontiguousarray(samples, dtype=np.float32),
        )
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise RuntimeError("Not enough speech for speaker verification")
        return np.asarray(self.extractor.compute(stream), dtype=np.float32)

    @staticmethod
    def _read_wav(path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError(f"Enrollment must be mono 16-bit PCM: {path}")
            sample_rate = source.getframerate()
            raw = source.readframes(source.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Enrollment must use {SAMPLE_RATE} Hz audio: {path}")
        return samples

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray) -> None:
        pcm = np.clip(samples, -1.0, 1.0)
        raw = (pcm * 32767.0).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(SAMPLE_RATE)
            target.writeframes(raw)

from __future__ import annotations

import collections
import datetime as dt
import html
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from conversation_catalog import ConversationCatalog
from faster_whisper import WhisperModel
from speaker_gate import SAMPLE_RATE, SpeakerGate

LOG = logging.getLogger("apartbrain-transcriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPTIONS_PATH = Path("/data/options.json")
OUTPUT_ROOT = Path("/share/apartbrain-conversations")
AUDIO_DIR = OUTPUT_ROOT / "audio"
TRANSCRIPT_DIR = OUTPUT_ROOT / "transcripts"
DIGEST_DIR = OUTPUT_ROOT / "digests"
STATUS_PATH = OUTPUT_ROOT / "status.json"
SPEAKER_DIR = OUTPUT_ROOT / "speakers"
SPEAKER_MODEL = Path("/opt/models/3dspeaker-campplus.onnx")
VAD_MODEL = Path("/opt/models/silero-vad.onnx")
WHISPER_MODEL_DIR = Path("/data/models")
SOURCE_SAMPLE_RATE = 48000
# Remove handling/room rumble and out-of-band noise, then raise quiet speech
# without clipping louder nearby voices. The retained FLAC stays unfiltered.
SPEECH_FILTER = "highpass=f=80,lowpass=f=7600,dynaudnorm=f=250:g=7:p=0.90:m=8"
STOP = threading.Event()
WAKE_TRANSCRIBER = threading.Event()
MODEL: WhisperModel | None = None
MODEL_LOCK = threading.Lock()
GATE: SpeakerGate | None = None
GATE_LOCK = threading.Lock()
SESSION_ACTIVE_UNTIL = 0.0
SESSION_TRIGGERED_BY: str | None = None


def load_options() -> dict:
    defaults = {
        "recording_enabled": False,
        "language": "auto",
        "model": "turbo",
        "segment_minutes": 30,
        "retention_days": 30,
        "digest_weekday": 0,
        "digest_hour": 9,
        "timezone": "Europe/Rome",
        "notify_services": ["mobile_app_pixel_10_pro"],
        "microphone_source": "default",
        "speaker_gate_enabled": True,
        "allowed_speakers": ["Vika", "Ale", "Andrey"],
        "speaker_match_threshold": 0.62,
        "speaker_confirmations": 2,
        "speaker_confirmation_window_seconds": 30,
        "conversation_hold_minutes": 5,
        "conversation_reply_window_seconds": 30,
        "vika_display_name": "Victoria",
        "unidentified_speaker_label": "Victoria",
        "transcription_prompt": (
            "Apartment family conversation in Russian, Italian, or English. "
            "Speaker names: Victoria, Ale, and Andrey. Preserve the spoken language."
        ),
    }
    try:
        defaults.update(json.loads(OPTIONS_PATH.read_text()))
    except FileNotFoundError:
        LOG.warning("No options file; using safe defaults with recording disabled")
    return defaults


OPTIONS = load_options()
CATALOG = ConversationCatalog(
    OUTPUT_ROOT,
    speaker_aliases={
        "Vika": str(OPTIONS["vika_display_name"]),
        "participant": str(OPTIONS["unidentified_speaker_label"]),
    },
)
RUNTIME_RECORDING = bool(OPTIONS["recording_enabled"]) and not bool(
    OPTIONS["speaker_gate_enabled"]
)
STATE_LOCK = threading.Lock()
STATUS = {
    "recording": False,
    "configured_enabled": bool(OPTIONS["recording_enabled"]),
    "last_audio": None,
    "last_transcript": None,
    "last_digest": None,
    "last_error": None,
    "gate_ready": False,
    "enrolled_speakers": [],
    "enrollment_target": None,
    "triggered_by": None,
    "catalog_database": str(CATALOG.database_path),
    "catalog_export_dir": str(CATALOG.export_dir),
    "catalog_conversations": 0,
    "catalog_transcribed": 0,
    "catalog_updated_at": None,
    "transcription_model": str(OPTIONS["model"]),
    "source_sample_rate": SOURCE_SAMPLE_RATE,
    "speech_enhancement": True,
}


def write_status(**updates) -> None:
    with STATE_LOCK:
        STATUS.update(updates)
        STATUS["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        temp = STATUS_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(STATUS, indent=2))
        temp.replace(STATUS_PATH)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def display_speaker(name: str | None) -> str:
    if name == "Vika":
        return str(OPTIONS["vika_display_name"])
    if name:
        return name
    return str(OPTIONS["unidentified_speaker_label"])


def refresh_catalog_status() -> None:
    summary = CATALOG.summary()
    write_status(
        catalog_database=str(CATALOG.database_path),
        catalog_export_dir=str(CATALOG.export_dir),
        catalog_conversations=summary["conversations"],
        catalog_transcribed=summary["transcribed"],
        catalog_updated_at=summary["last_updated"],
    )


def catalog_event(
    event_type: str,
    conversation_id: str | None = None,
    speaker: str | None = None,
    details: dict | None = None,
) -> None:
    try:
        CATALOG.record_event(
            event_type,
            conversation_id=conversation_id,
            speaker=display_speaker(speaker) if speaker else None,
            details=details,
        )
        refresh_catalog_status()
    except Exception as exc:
        LOG.exception("Conversation catalog event failed")
        write_status(last_error=f"catalog: {exc}")


def get_gate() -> SpeakerGate:
    global GATE
    with GATE_LOCK:
        if GATE is None:
            GATE = SpeakerGate(
                speaker_dir=SPEAKER_DIR,
                embedding_model=SPEAKER_MODEL,
                vad_model=VAD_MODEL,
                allowed_speakers=list(OPTIONS["allowed_speakers"]),
                threshold=float(OPTIONS["speaker_match_threshold"]),
            )
        return GATE


def speaker_monitor_loop() -> None:
    global RUNTIME_RECORDING, SESSION_ACTIVE_UNTIL, SESSION_TRIGGERED_BY
    if not bool(OPTIONS["speaker_gate_enabled"]):
        LOG.info("Speaker gate disabled; using recording_enabled directly")
        return

    RUNTIME_RECORDING = False
    confirmations: collections.deque[tuple[float, str]] = collections.deque()
    bytes_per_read = int(SAMPLE_RATE * 0.1) * 4
    last_status_write = 0.0
    last_gate_status: dict | None = None
    while not STOP.is_set():
        capture = None
        try:
            gate = get_gate()
            gate_status = gate.status()
            write_status(
                gate_ready=len(gate_status["enrolled_speakers"]) > 0,
                enrolled_speakers=[
                    display_speaker(name)
                    for name in gate_status["enrolled_speakers"]
                ],
                enrollment_target=(
                    display_speaker(gate_status["enrollment_target"])
                    if gate_status["enrollment_target"]
                    else None
                ),
            )
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "pulse",
                "-i",
                str(OPTIONS["microphone_source"]),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "f32le",
                "pipe:1",
            ]
            capture = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            LOG.info(
                "Speaker gate monitoring PulseAudio source %s in memory",
                OPTIONS["microphone_source"],
            )
            while not STOP.is_set() and capture.poll() is None:
                raw = capture.stdout.read(bytes_per_read) if capture.stdout else b""
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<f4").copy()
                now = time.monotonic()
                for name in gate.accept_monitor_audio(samples):
                    if RUNTIME_RECORDING:
                        SESSION_ACTIVE_UNTIL = (
                            now + int(OPTIONS["conversation_hold_minutes"]) * 60
                        )
                        SESSION_TRIGGERED_BY = name
                        write_status(
                            triggered_by=display_speaker(name), last_error=None
                        )
                        confirmations.clear()
                        continue

                    confirmations.append((now, name))
                    window = float(OPTIONS["speaker_confirmation_window_seconds"])
                    while confirmations and now - confirmations[0][0] > window:
                        confirmations.popleft()
                    matching = sum(
                        1 for _, candidate in confirmations if candidate == name
                    )
                    if matching >= int(OPTIONS["speaker_confirmations"]):
                        if bool(OPTIONS["recording_enabled"]):
                            first_trigger = not RUNTIME_RECORDING
                            RUNTIME_RECORDING = True
                            SESSION_ACTIVE_UNTIL = (
                                now + int(OPTIONS["conversation_hold_minutes"]) * 60
                            )
                            SESSION_TRIGGERED_BY = name
                            write_status(
                                triggered_by=display_speaker(name), last_error=None
                            )
                            if first_trigger:
                                LOG.info(
                                    "Conversation recording triggered by enrolled speaker %s",
                                    name,
                                )
                                catalog_event(
                                    "recording_triggered",
                                    speaker=name,
                                    details={"canonical_speaker": name},
                                )
                        confirmations.clear()

                if RUNTIME_RECORDING and now >= SESSION_ACTIVE_UNTIL:
                    ended_by = SESSION_TRIGGERED_BY
                    RUNTIME_RECORDING = False
                    SESSION_TRIGGERED_BY = None
                    write_status(triggered_by=None)
                    catalog_event("conversation_session_ended", speaker=ended_by)
                    LOG.info(
                        "Conversation session ended after enrolled-speaker inactivity"
                    )

                if now - last_status_write >= 2:
                    current = gate.status()
                    current_gate_status = {
                        "gate_ready": len(current["enrolled_speakers"]) > 0,
                        "enrolled_speakers": [
                            display_speaker(name)
                            for name in current["enrolled_speakers"]
                        ],
                        "enrollment_target": (
                            display_speaker(current["enrollment_target"])
                            if current["enrollment_target"]
                            else None
                        ),
                        "enrollment_progress": current["enrollment_progress"],
                    }
                    if current_gate_status != last_gate_status:
                        write_status(**current_gate_status)
                        last_gate_status = current_gate_status
                    last_status_write = now
            if capture.poll() not in (None, 0) and not STOP.is_set():
                raise RuntimeError(
                    f"speaker monitor ffmpeg exited with status {capture.returncode}"
                )
        except Exception as exc:
            write_status(last_error=f"speaker gate: {exc}")
            LOG.exception("Speaker gate failed")
            STOP.wait(5)
        finally:
            if capture and capture.poll() is None:
                capture.terminate()
                try:
                    capture.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    capture.kill()


def recording_loop() -> None:
    idle_reported = False
    while not STOP.is_set():
        if not RUNTIME_RECORDING:
            if not idle_reported:
                write_status(recording=False)
                idle_reported = True
            STOP.wait(2)
            continue
        idle_reported = False

        stamp = utc_stamp()
        partial = AUDIO_DIR / f"{stamp}.partial.flac"
        final = AUDIO_DIR / f"{stamp}.flac"
        trigger_speaker = SESSION_TRIGGERED_BY
        seconds = int(OPTIONS["segment_minutes"]) * 60
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "pulse",
            "-i",
            str(OPTIONS["microphone_source"]),
            "-ac",
            "1",
            "-ar",
            str(SOURCE_SAMPLE_RATE),
            "-t",
            str(seconds),
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            "-y",
            str(partial),
        ]
        write_status(recording=True, last_error=None)
        LOG.info(
            "Recording up to %s minutes, triggered by %s",
            OPTIONS["segment_minutes"],
            trigger_speaker or "manual mode",
        )
        started = time.monotonic()
        recorder = None
        try:
            recorder = subprocess.Popen(cmd)
            while recorder.poll() is None:
                if STOP.is_set() or not RUNTIME_RECORDING:
                    recorder.send_signal(signal.SIGINT)
                    try:
                        recorder.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        recorder.kill()
                    break
                STOP.wait(1)
            recorder.wait()
            duration = time.monotonic() - started
            if partial.is_file() and partial.stat().st_size > 1024 and duration >= 5:
                partial.replace(final)
                write_status(last_audio=final.name)
                try:
                    CATALOG.record_audio(
                        final,
                        display_speaker(trigger_speaker)
                        if trigger_speaker
                        else None,
                        duration,
                    )
                    refresh_catalog_status()
                except Exception as exc:
                    LOG.exception("Conversation catalog recording update failed")
                    write_status(last_error=f"catalog: {exc}")
                WAKE_TRANSCRIBER.set()
            else:
                partial.unlink(missing_ok=True)
            if recorder.returncode not in (0, 255) and not STOP.is_set():
                raise RuntimeError(f"ffmpeg exited with status {recorder.returncode}")
        except Exception as exc:
            partial.unlink(missing_ok=True)
            write_status(recording=False, last_error=f"recording: {exc}")
            LOG.exception("Recording failed")
            STOP.wait(10)
        finally:
            write_status(recording=False)
            if recorder and recorder.poll() is None:
                recorder.kill()


def get_model() -> WhisperModel:
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            LOG.info("Loading local Whisper model %s", OPTIONS["model"])
            WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            MODEL = WhisperModel(
                str(OPTIONS["model"]),
                device="cpu",
                compute_type="int8",
                download_root=str(WHISPER_MODEL_DIR),
                cpu_threads=max(1, min(4, os.cpu_count() or 2)),
            )
        return MODEL


def decode_audio_pcm(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            SPEECH_FILTER,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def transcribe_file(audio_path: Path) -> Path:
    language = None if OPTIONS["language"] == "auto" else str(OPTIONS["language"])
    prompt = str(OPTIONS["transcription_prompt"]).strip() or None
    # Decode once so transcription and speaker identification use the same
    # cleaned speech signal, while the original high-quality FLAC is retained.
    pcm = decode_audio_pcm(audio_path)
    segments, info = get_model().transcribe(
        pcm,
        language=language,
        initial_prompt=prompt,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        # Independent windows prevent a mistaken phrase near the end of a
        # noisy segment from feeding a repetition loop into later windows.
        condition_on_previous_text=False,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        word_timestamps=False,
    )
    candidates = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            candidates.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "speaker": None,
                }
            )

    rows = []
    filter_note = (
        "speaker gate enabled; no speech candidates"
        if bool(OPTIONS["speaker_gate_enabled"])
        else "speaker gate disabled"
    )
    if bool(OPTIONS["speaker_gate_enabled"]) and candidates:
        gate = get_gate()
        allowed = set(OPTIONS["allowed_speakers"])
        for candidate in candidates:
            start = max(0, int(candidate["start"] * SAMPLE_RATE))
            end = min(len(pcm), int(candidate["end"] * SAMPLE_RATE))
            candidate["speaker"] = gate.identify_samples(pcm[start:end])

        identified_anchors = [
            (candidate["start"] + candidate["end"]) / 2
            for candidate in candidates
            if candidate["speaker"] in allowed
        ]
        # Every persisted gated segment begins inside a verified conversation.
        # Keep its opening replies even though the two trigger utterances were
        # intentionally monitored only in memory and are not in the audio file.
        anchors = [0.0, *identified_anchors]
        reply_window = float(OPTIONS["conversation_reply_window_seconds"])
        kept = []
        for candidate in candidates:
            midpoint = (candidate["start"] + candidate["end"]) / 2
            near_anchor = any(
                abs(midpoint - anchor) <= reply_window for anchor in anchors
            )
            if candidate["speaker"] in allowed or near_anchor:
                kept.append(candidate)
        candidates = kept
        filter_note = (
            f"{len(candidates)} conversational turns kept; "
            f"{len(identified_anchors)} enrolled-speaker anchors plus session opening"
        )

    catalog_turns = []
    for candidate in candidates:
        speaker = display_speaker(candidate["speaker"])
        rows.append(
            f"[{format_offset(candidate['start'])}–{format_offset(candidate['end'])}] "
            f"[{speaker}] {candidate['text']}"
        )
        catalog_turns.append(
            {
                "start_seconds": candidate["start"],
                "end_seconds": candidate["end"],
                "speaker": speaker,
                "text": candidate["text"],
            }
        )

    try:
        created = dt.datetime.strptime(audio_path.stem, "%Y-%m-%dT%H-%M-%SZ").replace(
            tzinfo=dt.timezone.utc
        )
        created_label = created.isoformat()
    except ValueError:
        created_label = f"{audio_path.stem} UTC"
    body = (
        f"# Conversation transcript\n\n"
        f"- Audio segment: `{audio_path.name}`\n"
        f"- Transcription model: `{OPTIONS['model']}` (enhanced speech audio)\n"
        f"- Detected language: `{info.language}` ({info.language_probability:.0%})\n"
        f"- Recorded: `{created_label}`\n\n"
        f"- Conversation filter: `{filter_note}`\n\n"
        + (
            "\n\n".join(rows)
            if rows
            else "_No verified conversation detected; TV/background-only audio was discarded._"
        )
        + "\n"
    )
    output = TRANSCRIPT_DIR / f"{audio_path.stem}.md"
    temp = output.with_suffix(".tmp")
    temp.write_text(body, encoding="utf-8")
    temp.replace(output)
    catalog_ok = True
    try:
        CATALOG.record_transcript(
            audio_path,
            output,
            str(info.language),
            float(info.language_probability),
            filter_note,
            catalog_turns,
        )
        refresh_catalog_status()
    except Exception as exc:
        catalog_ok = False
        LOG.exception("Conversation catalog transcription update failed")
        write_status(last_error=f"catalog: {exc}")
    if catalog_ok:
        write_status(last_transcript=output.name, last_error=None)
    else:
        write_status(last_transcript=output.name)
    LOG.info("Transcribed %s -> %s", audio_path.name, output.name)
    return output


def format_offset(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def transcription_loop() -> None:
    while not STOP.is_set():
        try:
            done = {p.stem for p in TRANSCRIPT_DIR.glob("*.md")}
            pending = [
                p
                for p in sorted(AUDIO_DIR.glob("*.flac"))
                if not p.name.endswith(".partial.flac") and p.stem not in done
            ]
            for audio in pending:
                if STOP.is_set():
                    return
                transcribe_file(audio)
        except Exception as exc:
            write_status(last_error=f"transcription: {exc}")
            LOG.exception("Transcription failed")
        WAKE_TRANSCRIBER.wait(30)
        WAKE_TRANSCRIBER.clear()


def transcript_text(path: Path) -> str:
    text = path.read_text(errors="replace")
    return "\n".join(line for line in text.splitlines() if line.startswith("["))


def make_digest(days: int = 7, send: bool = True) -> tuple[Path, str]:
    cutoff = time.time() - days * 86400
    candidates = [
        p
        for p in sorted(TRANSCRIPT_DIR.glob("*.md"))
        if p.stat().st_mtime >= cutoff and not p.name.endswith(".partial.md")
    ]
    texts = [(p, transcript_text(p)) for p in candidates]
    texts = [(p, text) for p, text in texts if text]
    sources = [p for p, _ in texts]
    combined = "\n".join(text for _, text in texts)
    highlights = extract_highlights(combined, limit=10)
    generated = dt.datetime.now(ZoneInfo(str(OPTIONS["timezone"])))
    name = generated.strftime("%Y-%m-%d-weekly-digest.md")
    transcript_index = DIGEST_DIR / generated.strftime("%Y-%m-%d-full-transcript.md")

    full = [f"# Full transcript — {generated:%d %B %Y}\n"]
    for path, text in texts:
        full.append(f"\n## {path.stem}\n\n{text or '_No speech detected._'}\n")
    transcript_index.write_text("".join(full))

    digest = [
        f"# Conversation digest — {generated:%d %B %Y}\n",
        f"\nPeriod: previous {days} days · {len(sources)} recorded segments\n",
        "\n## Highlights\n",
    ]
    digest.extend(f"\n- {item}" for item in highlights)
    if not highlights:
        digest.append("\n- No transcribed speech was found for this period.")
    full_url = secure_ingress_url(f"transcript/{transcript_index.name}")
    digest.extend(
        [
            "\n\n## Full transcript\n",
            f"\n[Open the complete timestamped transcript]({full_url})\n",
        ]
    )
    digest_path = DIGEST_DIR / name
    digest_path.write_text("".join(digest))
    write_status(last_digest=digest_path.name)

    message = "\n".join(f"• {item}" for item in highlights[:5])
    if not message:
        message = "No transcribed speech was found this week."
    message += f"\n\nFull transcript: {full_url}"
    if send:
        notify("Apartment conversation digest", message, full_url)
    return digest_path, full_url


def extract_highlights(text: str, limit: int) -> list[str]:
    clean = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", text, flags=re.MULTILINE)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", clean)
        if len(s.strip().split()) >= 5
    ]
    stopwords = {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "because",
        "been",
        "before",
        "but",
        "can",
        "could",
        "for",
        "from",
        "have",
        "into",
        "just",
        "like",
        "that",
        "the",
        "their",
        "then",
        "there",
        "they",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
    }

    def words_in(value: str) -> list[str]:
        return re.findall(r"[^\W\d_]{3,}", value.lower(), flags=re.UNICODE)

    words = words_in(clean)
    freq = collections.Counter(word for word in words if word not in stopwords)
    scored = []
    for position, sentence in enumerate(sentences):
        tokens = words_in(sentence)
        score = sum(freq[t] for t in set(tokens) if t not in stopwords) / max(
            8, len(tokens)
        )
        scored.append((score, position, sentence))
    selected = sorted(scored, reverse=True)[:limit]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]


def notify(title: str, message: str, url: str) -> None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise RuntimeError("SUPERVISOR_TOKEN is unavailable")
    for service in OPTIONS["notify_services"]:
        payload = json.dumps(
            {
                "title": title,
                "message": message,
                "data": {"url": url, "clickAction": url},
            }
        ).encode()
        request = urllib.request.Request(
            f"http://supervisor/core/api/services/notify/{service}",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20):
                LOG.info("Sent digest through notify.%s", service)
        except urllib.error.HTTPError as exc:
            LOG.error(
                "notify.%s failed: %s %s",
                service,
                exc.code,
                exc.read().decode(errors="replace"),
            )


def supervisor_json(url: str) -> dict:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise RuntimeError("SUPERVISOR_TOKEN is unavailable")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def secure_ingress_url(relative_path: str) -> str:
    addon = supervisor_json("http://supervisor/addons/self/info").get("data", {})
    entry = str(addon.get("ingress_entry") or "").rstrip("/")
    if not entry:
        raise RuntimeError("Home Assistant did not provide an ingress URL")
    config = supervisor_json("http://supervisor/core/api/config")
    base = str(config.get("external_url") or config.get("internal_url") or "").rstrip(
        "/"
    )
    return (
        f"{base}{entry}/{relative_path.lstrip('/')}"
        if base
        else f"{entry}/{relative_path.lstrip('/')}"
    )


def cleanup_loop() -> None:
    while not STOP.wait(3600):
        cutoff = time.time() - int(OPTIONS["retention_days"]) * 86400
        for directory in (AUDIO_DIR, TRANSCRIPT_DIR, DIGEST_DIR):
            for path in directory.iterdir():
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)


def cleanup_stale_partials() -> None:
    for path in AUDIO_DIR.glob("*.partial.flac"):
        path.unlink(missing_ok=True)
    for path in TRANSCRIPT_DIR.glob("*.partial.md"):
        path.unlink(missing_ok=True)


def scheduler_loop() -> None:
    last_key = None
    zone = ZoneInfo(str(OPTIONS["timezone"]))
    while not STOP.wait(30):
        now = dt.datetime.now(zone)
        key = now.strftime("%G-%V")
        if (
            now.weekday() == int(OPTIONS["digest_weekday"])
            and now.hour == int(OPTIONS["digest_hour"])
            and key != last_key
        ):
            try:
                make_digest(7, send=True)
                last_key = key
            except Exception as exc:
                write_status(last_error=f"digest: {exc}")
                LOG.exception("Scheduled digest failed")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/api/status":
            return self.respond_json(STATUS)
        if route == "/api/catalog":
            return self.respond_json(
                {
                    "summary": CATALOG.summary(),
                    "recent": CATALOG.recent(100),
                    "database": str(CATALOG.database_path),
                    "exports": str(CATALOG.export_dir),
                }
            )
        if route == "/api/digest":
            path, url = make_digest(7, send=True)
            return self.respond_json(
                {"ok": True, "digest": path.name, "full_transcript": url}
            )
        if route == "/api/recording/start":
            return self.set_recording(True)
        if route == "/api/recording/stop":
            return self.set_recording(False)
        if route.startswith("/api/enroll/"):
            name = route.removeprefix("/api/enroll/")
            try:
                get_gate().start_enrollment(name)
                write_status(enrollment_target=display_speaker(name))
                return self.respond_json(
                    {"ok": True, "enrolling": display_speaker(name)}
                )
            except ValueError as exc:
                return self.respond_json({"ok": False, "error": str(exc)}, status=400)
        if route == "/api/enrollment/cancel":
            get_gate().cancel_enrollment()
            write_status(enrollment_target=None)
            return self.respond_json({"ok": True})
        if route.startswith("/transcript/"):
            return self.serve_transcript(route.removeprefix("/transcript/"))
        if route.startswith("/conversation-transcript/"):
            return self.serve_conversation_transcript(
                route.removeprefix("/conversation-transcript/")
            )
        if route.startswith("/recording/"):
            return self.serve_recording(route.removeprefix("/recording/"))
        if route.startswith("/catalog/"):
            return self.serve_catalog_file(route.removeprefix("/catalog/"))
        if route == "/":
            return self.respond_html(render_ui())
        self.send_error(404)

    def serve_transcript(self, filename: str):
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith(".md"):
            return self.send_error(400)
        path = DIGEST_DIR / safe_name
        if not path.is_file():
            return self.send_error(404)
        body = path.read_text(errors="replace")
        escaped = html.escape(body)
        return self.respond_html(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem;"
            "background:#101827;color:#eef}pre{white-space:pre-wrap;line-height:1.5}</style></head>"
            f"<body><pre>{escaped}</pre></body></html>"
        )

    def serve_conversation_transcript(self, filename: str):
        """Render one retained conversation transcript through protected ingress."""
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith(".md"):
            return self.send_error(400)
        path = TRANSCRIPT_DIR / safe_name
        if not path.is_file():
            return self.send_error(404)
        escaped = html.escape(path.read_text(encoding="utf-8", errors="replace"))
        return self.respond_html(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem;"
            "background:#101827;color:#eef}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
            "line-height:1.5}</style></head>"
            f"<body><pre>{escaped}</pre></body></html>"
        )

    def serve_recording(self, filename: str):
        """Stream one retained FLAC recording through protected ingress."""
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith(".flac"):
            return self.send_error(400)
        path = AUDIO_DIR / safe_name
        if not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "audio/flac")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def serve_catalog_file(self, filename: str):
        files = {
            "conversations.sqlite3": (
                CATALOG.export_dir / "conversations.sqlite3",
                "application/vnd.sqlite3",
            ),
            "conversations.csv": (
                CATALOG.export_dir / "conversations.csv",
                "text/csv; charset=utf-8",
            ),
            "turns.csv": (
                CATALOG.export_dir / "turns.csv",
                "text/csv; charset=utf-8",
            ),
            "events.csv": (
                CATALOG.export_dir / "events.csv",
                "text/csv; charset=utf-8",
            ),
            "conversations.jsonl": (
                CATALOG.export_dir / "conversations.jsonl",
                "application/x-ndjson; charset=utf-8",
            ),
        }
        if filename not in files:
            return self.send_error(404)
        CATALOG.export()
        path, content_type = files[filename]
        if not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Content-Disposition", f'attachment; filename="{path.name}"'
        )
        self.end_headers()
        self.wfile.write(data)

    def set_recording(self, enabled: bool):
        global RUNTIME_RECORDING
        if enabled and bool(OPTIONS["speaker_gate_enabled"]):
            return self.respond_json(
                {
                    "ok": False,
                    "error": "Speaker gate is enabled. Recording starts only after an enrolled speaker is verified.",
                },
                status=409,
            )
        RUNTIME_RECORDING = enabled
        write_status(configured_enabled=bool(OPTIONS["recording_enabled"]))
        return self.respond_json({"ok": True, "recording_requested": enabled})

    def respond_json(self, value: dict, status: int = 200):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_html(self, value: str):
        data = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        LOG.info("web: " + fmt, *args)


def render_ui() -> str:
    with STATE_LOCK:
        state = dict(STATUS)
    recording = "Recording" if state["recording"] else "Stopped"
    error = html.escape(str(state["last_error"] or "None"))
    enrolled = ", ".join(state.get("enrolled_speakers") or []) or "None"
    enrollment = html.escape(str(state.get("enrollment_target") or "None"))
    speaker_buttons = "".join(
        f"<button onclick=\"location.href='api/enroll/{html.escape(name)}'\">"
        f"I consent — enroll {html.escape(display_speaker(name))}</button>"
        for name in OPTIONS["allowed_speakers"]
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:16px system-ui;max-width:700px;margin:2rem auto;padding:1rem;background:#101827;color:#eef}}
button{{font:inherit;padding:.7rem 1rem;margin:.3rem;border:0;border-radius:.6rem}}a{{color:#7dd3fc}}code{{color:#fde68a}}</style></head>
<body><h1>Apartment conversation recorder</h1>
<p><strong>Status:</strong> {recording}</p>
<p><strong>Last audio:</strong> {html.escape(str(state["last_audio"] or "None"))}<br>
<strong>Last transcript:</strong> {html.escape(str(state["last_transcript"] or "None"))}<br>
<strong>Enrolled speakers:</strong> {html.escape(enrolled)}<br>
<strong>Enrollment in progress:</strong> {enrollment}<br>
<strong>Conversation triggered by:</strong> {html.escape(str(state.get("triggered_by") or "None"))}<br>
<strong>Last error:</strong> {error}</p>
<h2>Consent-based voice enrollment</h2>
<p><strong>Consent is personal:</strong> the person being enrolled must agree before their voice sample is captured.</p>
<ol><li>Ask the person to confirm that they consent to apartment conversation recording.</li>
<li>That person selects their own <strong>I consent — enroll</strong> button below.</li>
<li>They speak naturally near the microphone for about 10 seconds, until their name appears under Enrolled speakers.</li>
<li>After enrollment, recording starts automatically only when an enrolled speaker is verified twice during a real conversation.</li></ol>
{speaker_buttons}
<button onclick="location.href='api/enrollment/cancel'">Cancel enrollment</button>
<button onclick="location.href='api/recording/stop'">Stop recording</button>
<button onclick="location.href='api/digest'">Send digest now</button>
<h2>Local conversation catalogue</h2>
<p><strong>Indexed conversations:</strong> {int(state.get("catalog_conversations") or 0)}<br>
<strong>Transcribed:</strong> {int(state.get("catalog_transcribed") or 0)}<br>
<strong>Database on Pi:</strong> <code>{html.escape(str(state.get("catalog_database") or CATALOG.database_path))}</code></p>
<p><a href="catalog/conversations.sqlite3">Download portable SQLite database</a><br>
<a href="catalog/conversations.csv">Download conversations CSV</a><br>
<a href="catalog/turns.csv">Download timestamped turns CSV</a><br>
<a href="catalog/events.csv">Download recording/transcription events CSV</a><br>
<a href="catalog/conversations.jsonl">Download complete JSONL export</a></p>
<p>TV/background-only turns are discarded unless they occur near a verified enrolled-speaker turn. Transcripts are kept in the protected add-on share and linked from each digest. Place a visible notice and obtain consent from everyone who may be recorded.</p>
</body></html>"""


def shutdown(*_args) -> None:
    STOP.set()
    WAKE_TRANSCRIBER.set()


def main() -> None:
    for directory in (AUDIO_DIR, TRANSCRIPT_DIR, DIGEST_DIR, SPEAKER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cleanup_stale_partials()
    CATALOG.initialize(AUDIO_DIR, TRANSCRIPT_DIR)
    write_status()
    refresh_catalog_status()
    catalog_event(
        "app_started",
        details={
            "recording_enabled": bool(OPTIONS["recording_enabled"]),
            "speaker_gate_enabled": bool(OPTIONS["speaker_gate_enabled"]),
        },
    )
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    workers = [
        threading.Thread(target=speaker_monitor_loop, daemon=True),
        threading.Thread(target=recording_loop, daemon=True),
        threading.Thread(target=transcription_loop, daemon=True),
        threading.Thread(target=cleanup_loop, daemon=True),
        threading.Thread(target=scheduler_loop, daemon=True),
    ]
    for worker in workers:
        worker.start()
    server = ThreadingHTTPServer(("0.0.0.0", 8099), Handler)
    server.timeout = 1
    LOG.info(
        "Control panel listening on port 8099; recording configured=%s",
        RUNTIME_RECORDING,
    )
    while not STOP.is_set():
        server.handle_request()
    server.server_close()


if __name__ == "__main__":
    main()

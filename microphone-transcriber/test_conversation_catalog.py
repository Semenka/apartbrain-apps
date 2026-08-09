from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from conversation_catalog import ConversationCatalog, parse_transcript


SAMPLE_TRANSCRIPT = """# Conversation transcript

- Audio segment: `2026-08-09T10-14-24Z.flac`
- Detected language: `ru` (88%)
- Recorded: `2026-08-09T10:14:24+00:00`

- Conversation filter: `2 conversational turns kept; 1 enrolled-speaker anchors`

[00:00–00:02] [participant] Доброе утро.
[00:03–00:06] [Vika] Поедем на море.
"""


class ConversationCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.audio_dir = self.root / "audio"
        self.transcript_dir = self.root / "transcripts"
        self.audio_dir.mkdir()
        self.transcript_dir.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_parse_transcript(self) -> None:
        path = self.transcript_dir / "2026-08-09T10-14-24Z.md"
        path.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
        parsed = parse_transcript(path)
        self.assertEqual(parsed["language"], "ru")
        self.assertEqual(parsed["language_probability"], 0.88)
        self.assertEqual(parsed["turns"][0]["speaker"], "participant")
        self.assertEqual(parsed["turns"][1]["text"], "Поедем на море.")

    def test_backfill_normalizes_speakers_and_exports(self) -> None:
        audio = self.audio_dir / "2026-08-09T10-14-24Z.flac"
        audio.write_bytes(b"audio")
        transcript = self.transcript_dir / "2026-08-09T10-14-24Z.md"
        transcript.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")
        catalog = ConversationCatalog(
            self.root,
            speaker_aliases={"participant": "Victoria", "Vika": "Victoria"},
        )
        catalog.initialize(self.audio_dir, self.transcript_dir)

        with sqlite3.connect(catalog.database_path) as connection:
            row = connection.execute(
                "SELECT status, detected_language FROM conversations"
            ).fetchone()
            speakers = [
                item[0]
                for item in connection.execute(
                    "SELECT speaker FROM turns ORDER BY sequence"
                )
            ]
        self.assertEqual(row, ("transcribed", "ru"))
        self.assertEqual(speakers, ["Victoria", "Victoria"])
        self.assertTrue((catalog.export_dir / "conversations.sqlite3").is_file())
        self.assertTrue((catalog.export_dir / "conversations.csv").is_file())
        self.assertTrue((catalog.export_dir / "conversations.jsonl").is_file())

        with (catalog.export_dir / "turns.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["speaker"], "Victoria")

        exported = json.loads(
            (catalog.export_dir / "conversations.jsonl")
            .read_text(encoding="utf-8")
            .strip()
        )
        self.assertEqual(exported["turns"][1]["speaker"], "Victoria")

    def test_recording_and_transcription_events(self) -> None:
        catalog = ConversationCatalog(self.root)
        catalog.initialize(self.audio_dir, self.transcript_dir)
        audio = self.audio_dir / "2026-08-09T11-00-00Z.flac"
        audio.write_bytes(b"recording")
        transcript = self.transcript_dir / "2026-08-09T11-00-00Z.md"
        transcript.write_text("placeholder", encoding="utf-8")

        catalog.record_audio(audio, "Andrey", 12.5)
        catalog.record_transcript(
            audio,
            transcript,
            "ru",
            0.91,
            "1 conversational turn kept",
            [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "speaker": "Victoria",
                    "text": "Привет.",
                }
            ],
        )

        with sqlite3.connect(catalog.database_path) as connection:
            conversation = connection.execute(
                """
                SELECT trigger_speaker, duration_seconds, status
                FROM conversations
                """
            ).fetchone()
            event_types = {
                row[0] for row in connection.execute("SELECT event_type FROM events")
            }
        self.assertEqual(conversation, ("Andrey", 12.5, "transcribed"))
        self.assertEqual(
            event_types, {"recording_saved", "transcription_completed"}
        )


if __name__ == "__main__":
    unittest.main()

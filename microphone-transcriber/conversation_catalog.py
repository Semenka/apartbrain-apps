from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable


TRANSCRIPT_ROW = re.compile(
    r"^\[(?P<start>\d{2}:\d{2})–(?P<end>\d{2}:\d{2})\] "
    r"\[(?P<speaker>[^\]]+)\] (?P<text>.*)$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def recorded_at_from_stem(stem: str) -> str:
    try:
        return (
            dt.datetime.strptime(stem, "%Y-%m-%dT%H-%M-%SZ")
            .replace(tzinfo=dt.timezone.utc)
            .isoformat()
        )
    except ValueError:
        return utc_now()


def offset_seconds(value: str) -> float:
    minutes, seconds = value.split(":", 1)
    return float(int(minutes) * 60 + int(seconds))


def parse_transcript(path: Path) -> dict[str, Any]:
    body = path.read_text(encoding="utf-8", errors="replace")
    language_match = re.search(
        r"^- Detected language: `([^`]+)` \((\d+)%\)$", body, re.MULTILINE
    )
    filter_match = re.search(
        r"^- Conversation filter: `([^`]+)`$", body, re.MULTILINE
    )
    audio_match = re.search(r"^- Audio segment: `([^`]+)`$", body, re.MULTILINE)
    turns = []
    for line in body.splitlines():
        match = TRANSCRIPT_ROW.match(line)
        if not match:
            continue
        turns.append(
            {
                "start_seconds": offset_seconds(match.group("start")),
                "end_seconds": offset_seconds(match.group("end")),
                "speaker": match.group("speaker"),
                "text": match.group("text"),
            }
        )
    return {
        "audio_filename": (
            audio_match.group(1) if audio_match else f"{path.stem}.flac"
        ),
        "language": language_match.group(1) if language_match else None,
        "language_probability": (
            int(language_match.group(2)) / 100 if language_match else None
        ),
        "filter_note": filter_match.group(1) if filter_match else None,
        "turns": turns,
        "transcript_text": "\n".join(
            line for line in body.splitlines() if TRANSCRIPT_ROW.match(line)
        ),
    }


class ConversationCatalog:
    """Portable SQLite index plus CSV/JSONL exports for local conversations."""

    def __init__(
        self,
        root: Path,
        speaker_aliases: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.database_path = root / "conversations.sqlite3"
        self.export_dir = root / "exports"
        self.speaker_aliases = speaker_aliases or {}
        self.lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self, audio_dir: Path, transcript_dir: Path) -> None:
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.export_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        recorded_at_utc TEXT NOT NULL,
                        trigger_speaker TEXT,
                        audio_filename TEXT,
                        audio_bytes INTEGER,
                        duration_seconds REAL,
                        transcript_filename TEXT,
                        detected_language TEXT,
                        language_probability REAL,
                        filter_note TEXT,
                        transcript_text TEXT,
                        status TEXT NOT NULL DEFAULT 'recorded',
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_conversations_recorded_at
                        ON conversations(recorded_at_utc);
                    CREATE INDEX IF NOT EXISTS idx_conversations_trigger_speaker
                        ON conversations(trigger_speaker);

                    CREATE TABLE IF NOT EXISTS turns (
                        conversation_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL,
                        speaker TEXT NOT NULL,
                        text TEXT NOT NULL,
                        PRIMARY KEY (conversation_id, sequence),
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_turns_speaker
                        ON turns(speaker);

                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        conversation_id TEXT,
                        speaker TEXT,
                        details_json TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_events_timestamp
                        ON events(timestamp_utc);
                    CREATE INDEX IF NOT EXISTS idx_events_type
                        ON events(event_type);
                    """
                )
                self._backfill(connection, audio_dir, transcript_dir)
            self.export()

    def _backfill(
        self,
        connection: sqlite3.Connection,
        audio_dir: Path,
        transcript_dir: Path,
    ) -> None:
        now = utc_now()
        for audio_path in sorted(audio_dir.glob("*.flac")):
            connection.execute(
                """
                INSERT INTO conversations (
                    id, recorded_at_utc, audio_filename, audio_bytes,
                    status, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, 'recorded', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    audio_filename = excluded.audio_filename,
                    audio_bytes = excluded.audio_bytes,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    audio_path.stem,
                    recorded_at_from_stem(audio_path.stem),
                    audio_path.name,
                    audio_path.stat().st_size,
                    now,
                    now,
                ),
            )

        for transcript_path in sorted(transcript_dir.glob("*.md")):
            parsed = parse_transcript(transcript_path)
            for turn in parsed["turns"]:
                turn["speaker"] = self.speaker_aliases.get(
                    str(turn["speaker"]), str(turn["speaker"])
                )
            parsed["transcript_text"] = "\n".join(
                f"[{turn['speaker']}] {turn['text']}" for turn in parsed["turns"]
            )
            conversation_id = transcript_path.stem
            connection.execute(
                """
                INSERT INTO conversations (
                    id, recorded_at_utc, audio_filename, transcript_filename,
                    detected_language, language_probability, filter_note,
                    transcript_text, status, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'transcribed', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    audio_filename = COALESCE(
                        conversations.audio_filename, excluded.audio_filename
                    ),
                    transcript_filename = excluded.transcript_filename,
                    detected_language = excluded.detected_language,
                    language_probability = excluded.language_probability,
                    filter_note = excluded.filter_note,
                    transcript_text = excluded.transcript_text,
                    status = 'transcribed',
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    conversation_id,
                    recorded_at_from_stem(conversation_id),
                    parsed["audio_filename"],
                    transcript_path.name,
                    parsed["language"],
                    parsed["language_probability"],
                    parsed["filter_note"],
                    parsed["transcript_text"],
                    now,
                    now,
                ),
            )
            self._replace_turns(connection, conversation_id, parsed["turns"])

    def record_event(
        self,
        event_type: str,
        conversation_id: str | None = None,
        speaker: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    timestamp_utc, event_type, conversation_id, speaker, details_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    event_type,
                    conversation_id,
                    speaker,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def record_audio(
        self,
        audio_path: Path,
        trigger_speaker: str | None,
        duration_seconds: float,
    ) -> None:
        now = utc_now()
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, recorded_at_utc, trigger_speaker, audio_filename,
                    audio_bytes, duration_seconds, status,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'recorded', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    trigger_speaker = excluded.trigger_speaker,
                    audio_filename = excluded.audio_filename,
                    audio_bytes = excluded.audio_bytes,
                    duration_seconds = excluded.duration_seconds,
                    status = CASE
                        WHEN conversations.transcript_filename IS NULL
                        THEN 'recorded' ELSE conversations.status END,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    audio_path.stem,
                    recorded_at_from_stem(audio_path.stem),
                    trigger_speaker,
                    audio_path.name,
                    audio_path.stat().st_size,
                    round(duration_seconds, 3),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO events (
                    timestamp_utc, event_type, conversation_id, speaker, details_json
                ) VALUES (?, 'recording_saved', ?, ?, ?)
                """,
                (
                    now,
                    audio_path.stem,
                    trigger_speaker,
                    json.dumps(
                        {
                            "audio_filename": audio_path.name,
                            "audio_bytes": audio_path.stat().st_size,
                            "duration_seconds": round(duration_seconds, 3),
                        },
                        sort_keys=True,
                    ),
                ),
            )
        self.export()

    def record_transcript(
        self,
        audio_path: Path,
        transcript_path: Path,
        language: str,
        language_probability: float,
        filter_note: str,
        turns: Iterable[dict[str, Any]],
    ) -> None:
        turn_list = list(turns)
        transcript_text = "\n".join(
            f"[{row['speaker']}] {row['text']}" for row in turn_list
        )
        now = utc_now()
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, recorded_at_utc, audio_filename, transcript_filename,
                    detected_language, language_probability, filter_note,
                    transcript_text, status, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'transcribed', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    audio_filename = excluded.audio_filename,
                    transcript_filename = excluded.transcript_filename,
                    detected_language = excluded.detected_language,
                    language_probability = excluded.language_probability,
                    filter_note = excluded.filter_note,
                    transcript_text = excluded.transcript_text,
                    status = 'transcribed',
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    audio_path.stem,
                    recorded_at_from_stem(audio_path.stem),
                    audio_path.name,
                    transcript_path.name,
                    language,
                    float(language_probability),
                    filter_note,
                    transcript_text,
                    now,
                    now,
                ),
            )
            self._replace_turns(connection, audio_path.stem, turn_list)
            connection.execute(
                """
                INSERT INTO events (
                    timestamp_utc, event_type, conversation_id, details_json
                ) VALUES (?, 'transcription_completed', ?, ?)
                """,
                (
                    now,
                    audio_path.stem,
                    json.dumps(
                        {
                            "transcript_filename": transcript_path.name,
                            "detected_language": language,
                            "language_probability": float(language_probability),
                            "turn_count": len(turn_list),
                        },
                        sort_keys=True,
                    ),
                ),
            )
        self.export()

    @staticmethod
    def _replace_turns(
        connection: sqlite3.Connection,
        conversation_id: str,
        turns: Iterable[dict[str, Any]],
    ) -> None:
        connection.execute(
            "DELETE FROM turns WHERE conversation_id = ?", (conversation_id,)
        )
        connection.executemany(
            """
            INSERT INTO turns (
                conversation_id, sequence, start_seconds, end_seconds, speaker, text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    conversation_id,
                    index,
                    float(row["start_seconds"]),
                    float(row["end_seconds"]),
                    str(row["speaker"]),
                    str(row["text"]),
                )
                for index, row in enumerate(turns, start=1)
            ],
        )

    def summary(self) -> dict[str, Any]:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS conversations,
                    SUM(CASE WHEN status = 'transcribed' THEN 1 ELSE 0 END)
                        AS transcribed,
                    MAX(updated_at_utc) AS last_updated
                FROM conversations
                """
            ).fetchone()
        return {
            "conversations": int(row["conversations"] or 0),
            "transcribed": int(row["transcribed"] or 0),
            "last_updated": row["last_updated"],
        }

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id, recorded_at_utc, trigger_speaker, audio_filename,
                    audio_bytes, duration_seconds, transcript_filename,
                    detected_language, language_probability, filter_note,
                    status, updated_at_utc
                FROM conversations
                ORDER BY recorded_at_utc DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def export(self) -> None:
        # HTTP requests can ask for different catalogue files at the same time.
        # Keep the lock for the complete export so those requests never share or
        # replace the same temporary CSV, JSONL, and SQLite snapshot files.
        with self.lock:
            with self._connect() as connection:
                conversations = connection.execute(
                    "SELECT * FROM conversations ORDER BY recorded_at_utc"
                ).fetchall()
                turns = connection.execute(
                    "SELECT * FROM turns ORDER BY conversation_id, sequence"
                ).fetchall()
                events = connection.execute(
                    "SELECT * FROM events ORDER BY timestamp_utc, id"
                ).fetchall()

            self._write_csv("conversations.csv", conversations)
            self._write_csv("turns.csv", turns)
            self._write_csv("events.csv", events)
            self._write_jsonl(conversations, turns)
            self._write_snapshot()

    def _write_csv(self, filename: str, rows: list[sqlite3.Row]) -> None:
        path = self.export_dir / filename
        temp = path.with_suffix(path.suffix + ".tmp")
        fieldnames = list(rows[0].keys()) if rows else []
        with temp.open("w", encoding="utf-8", newline="") as target:
            if fieldnames:
                writer = csv.DictWriter(target, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dict(row) for row in rows)
        temp.replace(path)

    def _write_jsonl(
        self,
        conversations: list[sqlite3.Row],
        turns: list[sqlite3.Row],
    ) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in turns:
            item = dict(row)
            grouped.setdefault(str(item.pop("conversation_id")), []).append(item)
        path = self.export_dir / "conversations.jsonl"
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as target:
            for row in conversations:
                item = dict(row)
                item["turns"] = grouped.get(str(item["id"]), [])
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp.replace(path)

    def _write_snapshot(self) -> None:
        path = self.export_dir / "conversations.sqlite3"
        temp = self.export_dir / "conversations.sqlite3.tmp"
        temp.unlink(missing_ok=True)
        with self._connect() as source, sqlite3.connect(temp) as target:
            source.backup(target)
        temp.replace(path)

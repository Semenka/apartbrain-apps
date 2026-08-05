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

from faster_whisper import WhisperModel

LOG = logging.getLogger("apartbrain-transcriber")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPTIONS_PATH = Path("/data/options.json")
OUTPUT_ROOT = Path("/share/apartbrain-conversations")
AUDIO_DIR = OUTPUT_ROOT / "audio"
TRANSCRIPT_DIR = OUTPUT_ROOT / "transcripts"
DIGEST_DIR = OUTPUT_ROOT / "digests"
STATUS_PATH = OUTPUT_ROOT / "status.json"
STOP = threading.Event()
WAKE_TRANSCRIBER = threading.Event()
MODEL: WhisperModel | None = None
MODEL_LOCK = threading.Lock()


def load_options() -> dict:
    defaults = {
        "recording_enabled": False,
        "language": "auto",
        "model": "small",
        "segment_minutes": 15,
        "retention_days": 30,
        "digest_weekday": 0,
        "digest_hour": 9,
        "timezone": "Europe/Rome",
        "notify_services": ["mobile_app_YOUR_PHONE"],
        "microphone_source": "default",
    }
    try:
        defaults.update(json.loads(OPTIONS_PATH.read_text()))
    except FileNotFoundError:
        LOG.warning("No options file; using safe defaults with recording disabled")
    return defaults


OPTIONS = load_options()
RUNTIME_RECORDING = bool(OPTIONS["recording_enabled"])
STATE_LOCK = threading.Lock()
STATUS = {
    "recording": False,
    "configured_enabled": bool(OPTIONS["recording_enabled"]),
    "last_audio": None,
    "last_transcript": None,
    "last_digest": None,
    "last_error": None,
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


def recording_loop() -> None:
    global RUNTIME_RECORDING
    while not STOP.is_set():
        if not RUNTIME_RECORDING:
            write_status(recording=False)
            STOP.wait(2)
            continue

        stamp = utc_stamp()
        partial = AUDIO_DIR / f"{stamp}.partial.flac"
        final = AUDIO_DIR / f"{stamp}.flac"
        seconds = int(OPTIONS["segment_minutes"]) * 60
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "pulse", "-i", str(OPTIONS["microphone_source"]),
            "-ac", "1", "-ar", "16000", "-t", str(seconds),
            "-c:a", "flac", "-y", str(partial),
        ]
        write_status(recording=True, last_error=None)
        LOG.info("Recording %s-minute segment from PulseAudio source %s", OPTIONS["segment_minutes"], OPTIONS["microphone_source"])
        try:
            result = subprocess.run(cmd, timeout=seconds + 45, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg exited with status {result.returncode}")
            partial.replace(final)
            write_status(last_audio=final.name)
            WAKE_TRANSCRIBER.set()
        except Exception as exc:
            partial.unlink(missing_ok=True)
            write_status(recording=False, last_error=f"recording: {exc}")
            LOG.exception("Recording failed")
            STOP.wait(10)


def get_model() -> WhisperModel:
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            LOG.info("Loading local Whisper model %s", OPTIONS["model"])
            MODEL = WhisperModel(str(OPTIONS["model"]), device="cpu", compute_type="int8")
        return MODEL


def transcribe_file(audio_path: Path) -> Path:
    language = None if OPTIONS["language"] == "auto" else str(OPTIONS["language"])
    segments, info = get_model().transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    rows = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            rows.append(f"[{format_offset(segment.start)}–{format_offset(segment.end)}] {text}")
    created = audio_path.stem.replace("T", " ").replace("-", ":", 2)
    body = (
        f"# Conversation transcript\n\n"
        f"- Audio segment: `{audio_path.name}`\n"
        f"- Detected language: `{info.language}` ({info.language_probability:.0%})\n"
        f"- Recorded: `{created} UTC`\n\n"
        + ("\n\n".join(rows) if rows else "_No speech detected._")
        + "\n"
    )
    output = TRANSCRIPT_DIR / f"{audio_path.stem}.md"
    temp = output.with_suffix(".tmp")
    temp.write_text(body)
    temp.replace(output)
    write_status(last_transcript=output.name, last_error=None)
    LOG.info("Transcribed %s -> %s", audio_path.name, output.name)
    return output


def format_offset(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def transcription_loop() -> None:
    while not STOP.is_set():
        try:
            done = {p.stem for p in TRANSCRIPT_DIR.glob("*.md")}
            pending = [p for p in sorted(AUDIO_DIR.glob("*.flac")) if p.stem not in done]
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
    sources = [p for p in sorted(TRANSCRIPT_DIR.glob("*.md")) if p.stat().st_mtime >= cutoff]
    texts = [(p, transcript_text(p)) for p in sources]
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
    digest.extend([
        "\n\n## Full transcript\n",
        f"\n[Open the complete timestamped transcript]({full_url})\n",
    ])
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
    clean = re.sub(r"^\[[^\]]+\]\s*", "", text, flags=re.MULTILINE)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.strip().split()) >= 5]
    stopwords = {
        "about", "after", "again", "also", "and", "are", "because", "been", "before",
        "but", "can", "could", "for", "from", "have", "into", "just", "like", "that",
        "the", "their", "then", "there", "they", "this", "was", "were", "what", "when",
        "where", "which", "will", "with", "would", "you", "your",
    }
    words = re.findall(r"[A-Za-zÀ-ÿ']{3,}", clean.lower())
    freq = collections.Counter(word for word in words if word not in stopwords)
    scored = []
    for position, sentence in enumerate(sentences):
        tokens = re.findall(r"[A-Za-zÀ-ÿ']{3,}", sentence.lower())
        score = sum(freq[t] for t in set(tokens) if t not in stopwords) / max(8, len(tokens))
        scored.append((score, position, sentence))
    selected = sorted(scored, reverse=True)[:limit]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]


def notify(title: str, message: str, url: str) -> None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise RuntimeError("SUPERVISOR_TOKEN is unavailable")
    for service in OPTIONS["notify_services"]:
        payload = json.dumps({
            "title": title,
            "message": message,
            "data": {"url": url, "clickAction": url},
        }).encode()
        request = urllib.request.Request(
            f"http://supervisor/core/api/services/notify/{service}",
            data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20):
                LOG.info("Sent digest through notify.%s", service)
        except urllib.error.HTTPError as exc:
            LOG.error("notify.%s failed: %s %s", service, exc.code, exc.read().decode(errors="replace"))


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
    base = str(config.get("external_url") or config.get("internal_url") or "").rstrip("/")
    return f"{base}{entry}/{relative_path.lstrip('/')}" if base else f"{entry}/{relative_path.lstrip('/')}"


def cleanup_loop() -> None:
    while not STOP.wait(3600):
        cutoff = time.time() - int(OPTIONS["retention_days"]) * 86400
        for directory in (AUDIO_DIR, TRANSCRIPT_DIR, DIGEST_DIR):
            for path in directory.iterdir():
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)


def scheduler_loop() -> None:
    last_key = None
    zone = ZoneInfo(str(OPTIONS["timezone"]))
    while not STOP.wait(30):
        now = dt.datetime.now(zone)
        key = now.strftime("%G-%V")
        if now.weekday() == int(OPTIONS["digest_weekday"]) and now.hour == int(OPTIONS["digest_hour"]) and key != last_key:
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
        if route == "/api/digest":
            path, url = make_digest(7, send=True)
            return self.respond_json({"ok": True, "digest": path.name, "full_transcript": url})
        if route == "/api/recording/start":
            return self.set_recording(True)
        if route == "/api/recording/stop":
            return self.set_recording(False)
        if route.startswith("/transcript/"):
            return self.serve_transcript(route.removeprefix("/transcript/"))
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
            "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem;"
            "background:#101827;color:#eef}pre{white-space:pre-wrap;line-height:1.5}</style></head>"
            f"<body><pre>{escaped}</pre></body></html>"
        )

    def set_recording(self, enabled: bool):
        global RUNTIME_RECORDING
        RUNTIME_RECORDING = enabled
        write_status(configured_enabled=bool(OPTIONS["recording_enabled"]))
        return self.respond_json({"ok": True, "recording_requested": enabled})

    def respond_json(self, value: dict):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_html(self, value: str):
        data = value.encode()
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
    return f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:16px system-ui;max-width:700px;margin:2rem auto;padding:1rem;background:#101827;color:#eef}}
button{{font:inherit;padding:.7rem 1rem;margin:.3rem;border:0;border-radius:.6rem}}a{{color:#7dd3fc}}code{{color:#fde68a}}</style></head>
<body><h1>Apartment conversation recorder</h1>
<p><strong>Status:</strong> {recording}</p>
<p><strong>Last audio:</strong> {html.escape(str(state["last_audio"] or "None"))}<br>
<strong>Last transcript:</strong> {html.escape(str(state["last_transcript"] or "None"))}<br>
<strong>Last error:</strong> {error}</p>
<button onclick="location.href='api/recording/start'">Start recording</button>
<button onclick="location.href='api/recording/stop'">Stop recording</button>
<button onclick="location.href='api/digest'">Send digest now</button>
<p>Transcripts are kept in the protected add-on share and linked from each digest. Recording people without notice may be restricted; place a visible notice and obtain consent.</p>
</body></html>"""


def shutdown(*_args) -> None:
    STOP.set()
    WAKE_TRANSCRIBER.set()


def main() -> None:
    for directory in (AUDIO_DIR, TRANSCRIPT_DIR, DIGEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    write_status()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    workers = [
        threading.Thread(target=recording_loop, daemon=True),
        threading.Thread(target=transcription_loop, daemon=True),
        threading.Thread(target=cleanup_loop, daemon=True),
        threading.Thread(target=scheduler_loop, daemon=True),
    ]
    for worker in workers:
        worker.start()
    server = ThreadingHTTPServer(("0.0.0.0", 8099), Handler)
    server.timeout = 1
    LOG.info("Control panel listening on port 8099; recording configured=%s", RUNTIME_RECORDING)
    while not STOP.is_set():
        server.handle_request()
    server.server_close()


if __name__ == "__main__":
    main()

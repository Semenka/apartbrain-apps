# Apartbrain Conversation Transcriber

This Home Assistant app records a USB microphone attached to the Raspberry Pi,
transcribes speech locally, and sends a weekly digest with a link to the full
timestamped transcript.

## Privacy first

- Audio never leaves the Pi for transcription.
- Recording is disabled on first installation.
- Put a visible notice near the recorded area and obtain consent from everyone
  who may be recorded.
- The default retention period is 30 days.
- Transcript links use Home Assistant's authenticated app ingress.

## Install

1. Confirm the microphone appears in **Settings → System → Hardware → Audio**.
2. Add `https://github.com/Semenka/apartbrain` as a Home Assistant app repository.
3. Install **Apartbrain Conversation Transcriber**.
4. Start it once with `recording_enabled: false`; open its web UI and confirm
   there is no microphone error.
5. Set `recording_enabled: true` and restart the app.
6. Speak for one recording segment, then verify that `last_audio` and
   `last_transcript` advance in the app web UI.

The first transcription downloads the selected Whisper model and can take
several minutes. `small` is the default multilingual model. Use `base` if the
Pi is too slow.

## Weekly and on-demand delivery

`digest_weekday: 0` means Monday. The default sends at 09:00 Europe/Rome using:

```yaml
notify_services:
  - mobile_app_YOUR_PHONE
```

Each item is the suffix of a Home Assistant `notify.*` service. Add an email
notification service here when one is configured.

To request a digest immediately, open the app web UI and select **Send digest
now**. The JSON endpoint is also `GET /api/digest`.

## Files

- Audio: `/share/apartbrain-conversations/audio/`
- Transcripts: `/share/apartbrain-conversations/transcripts/`
- Digests: `/share/apartbrain-conversations/digests/`
- Status: `/share/apartbrain-conversations/status.json`

The weekly message links to the full transcript through the app's authenticated
Home Assistant ingress. The files are not published under `/local/`.

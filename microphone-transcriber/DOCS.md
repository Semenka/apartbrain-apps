# Apartbrain Conversation Transcriber

This Home Assistant app monitors a USB microphone attached to the Raspberry Pi
in memory, starts persistent recording only after an enrolled speaker is
verified, transcribes locally, and produces weekly/on-demand digests with a
link to the full timestamped transcript.

## Privacy first

- Audio never leaves the Pi for transcription.
- Recording is disabled on first installation.
- In-memory monitoring is not saved. Persistent recording starts only after the
  same enrolled speaker is verified twice inside the confirmation window.
- Voice enrollment is opt-in and stored only on the Pi.
- Put a visible notice near the recorded area and obtain consent from everyone
  who may be recorded.
- The default retention period is 30 days.
- Transcript links use Home Assistant's authenticated app ingress.

## Install

1. Confirm the microphone appears in **Settings → System → Hardware → Audio**.
2. Add `https://github.com/Semenka/apartbrain-apps` as a Home Assistant app repository.
3. Install **Apartbrain Conversation Transcriber**.
4. Start it once with `recording_enabled: false`; open its web UI and confirm
   there is no microphone error.
5. With each person's explicit consent, select **Enroll Vika**, **Enroll Ale**,
   or **Enroll Andrey**, then have that person speak naturally for about
   10 seconds. The web UI shows completed enrollments.
6. Set `recording_enabled: true` and restart the app. The configured maximum
   segment length is 30 minutes.
7. Have an enrolled person initiate a real conversation, then verify that
   `triggered_by`, `last_audio`, and
   `last_transcript` advance in the app web UI.

The first transcription downloads the selected Whisper model and can take
several minutes. `small` is the default multilingual model. Speaker
verification and VAD use local ONNX models and work offline after installation.

## Conversation and TV filtering

- Two verified turns from an enrolled speaker are required to open a recording
  session.
- A session closes after five minutes without another verified enrolled-speaker
  turn. Files rotate at a maximum of 30 minutes.
- During transcription, only enrolled-speaker turns and nearby reply turns are
  retained. Speech separated from an enrolled-speaker anchor is classified as
  TV/background and omitted.
- A single microphone cannot provide perfect acoustic source separation. Keep
  the match threshold conservative and review the first week of results.

## Weekly and on-demand delivery

`digest_weekday: 0` means Monday. The default sends at 09:00 Europe/Rome using:

```yaml
notify_services:
  - mobile_app_pixel_10_pro
```

Each item is the suffix of a Home Assistant `notify.*` service. Email delivery
can be scheduled separately so no email password needs to be stored on the Pi.

To request a digest immediately, open the app web UI and select **Send digest
now**. The JSON endpoint is also `GET /api/digest`.

## Files

- Audio: `/share/apartbrain-conversations/audio/`
- Transcripts: `/share/apartbrain-conversations/transcripts/`
- Digests: `/share/apartbrain-conversations/digests/`
- Consented speaker samples: `/share/apartbrain-conversations/speakers/`
- Status: `/share/apartbrain-conversations/status.json`

The weekly message links to the full transcript through the app's authenticated
Home Assistant ingress. The files are not published under `/local/`.

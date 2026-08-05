ARG BUILD_FROM=python:3.11-slim-bookworm
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "faster-whisper==1.2.1" "sherpa-onnx==1.13.4" \
    && mkdir -p /opt/models \
    && curl -fsSL -o /opt/models/3dspeaker-campplus.onnx \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
    && curl -fsSL -o /opt/models/silero-vad.onnx \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

COPY app.py /app/app.py
COPY speaker_gate.py /app/speaker_gate.py
COPY run.sh /run.sh
RUN chmod 0755 /run.sh

CMD ["/run.sh"]

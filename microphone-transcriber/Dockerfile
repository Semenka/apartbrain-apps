ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "faster-whisper==1.2.1"

COPY app.py /app/app.py
COPY run.sh /run.sh
RUN chmod 0755 /run.sh

CMD ["/run.sh"]


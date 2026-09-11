# Bolt API, dashboard and workers.
#
# The always-on voice loop stays on the host: a container on macOS has no
# microphone or speaker. But the dashboard's mic button records in the browser and
# transcribes server-side, so whisper.cpp is built in here too — otherwise that
# button would be dead in the dockerised deployment.

# Build whisper.cpp separately so its toolchain does not bloat the final image.
# GGML_NATIVE=OFF matters: native detection emits an -mcpu value that the gcc in
# bookworm rejects outright, and it would tune for the build machine anyway.
FROM debian:bookworm-slim AS whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.7.4 \
        https://github.com/ggerganov/whisper.cpp /src \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
        -DGGML_NATIVE=OFF \
    && cmake --build /src/build --config Release -j "$(nproc)" --target whisper-cli

FROM python:3.13-slim

# ffmpeg for video frame extraction; git for the code gate's snapshots.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 rather than the pip-installable v1, so profiles and SSO behave the
# same way they do on the host.
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscli.zip \
    && apt-get update && apt-get install -y --no-install-recommends unzip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws \
    && apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# The Docker CLI, so the sandbox can use the host daemon through a mounted socket.
RUN curl -fsSL https://download.docker.com/linux/static/stable/aarch64/docker-27.4.0.tgz \
        -o /tmp/d.tgz \
    && tar -xzf /tmp/d.tgz -C /tmp \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/d.tgz /tmp/docker

# whisper-cli plus the shared libraries it links against.
COPY --from=whisper /src/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper /src/build/src/libwhisper.so* /usr/local/lib/
COPY --from=whisper /src/build/ggml/src/libggml*.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app

COPY pyproject.toml README.md ./
COPY bolt ./bolt
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir pytest boto3

# Model files and the workspace are mounted at run time, not baked in.
ENV BOLT_DB=/data/bolt.db \
    BOLT_WORKSPACE=/workspace \
    BOLT_SNAPSHOT_ROOT=/data/worktrees \
    BOLT_MODELS_DIR=/models \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["bolt", "serve", "--host", "0.0.0.0", "--port", "8080"]

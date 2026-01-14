ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=jammy-20250415.1

# -------- Stage 1: Build Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

WORKDIR /build

COPY uv.lock pyproject.toml README.md LICENSE ./
COPY ./src ./src

RUN uv build
    
# -------- Stage 2: Python Environment Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS python-env

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.13
RUN apt update && \
    apt install -y curl software-properties-common libgl1 libglib2.0-0 libmagic1 file build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.13 python3.13-dev && \
    apt-get clean

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /build/pyproject.toml /workspace/
COPY --from=builder /build/uv.lock /workspace/

RUN uv sync --locked --no-install-project --no-dev --extra ingest --extra elasticsearch

COPY --from=builder /build/dist/*.whl /workspace/

# Find the exact wheel name with version and store it in a variable
RUN WHEEL_NAME=$(ls /workspace/nvidia_rag-*.whl) && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME"

# Build arguments for optional tokenizer pre-download
# Example: docker build --build-arg PREDOWNLOAD_TOKENIZER=intfloat/e5-large-unsupervised \
#                       --build-arg HF_TOKEN=your_hf_token ...
ARG MODEL_PREDOWNLOAD_PATH=/workspace/models
ARG PREDOWNLOAD_TOKENIZER=
ARG HF_TOKEN=
ENV MODEL_PREDOWNLOAD_PATH=${MODEL_PREDOWNLOAD_PATH}
ENV HF_HOME=${MODEL_PREDOWNLOAD_PATH}
ENV PREDOWNLOAD_TOKENIZER=${PREDOWNLOAD_TOKENIZER}
ENV HF_TOKEN=${HF_TOKEN}

# Ensure models directory exists
RUN mkdir -p ${MODEL_PREDOWNLOAD_PATH}

# Copy post build triggers script
COPY ./src/nvidia_rag/ingestor_server/docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

# Pre-download tokenizer if PREDOWNLOAD_TOKENIZER is set
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$PREDOWNLOAD_TOKENIZER" ]; then \
        /workspace/.venv/bin/python /workspace/docker/post_build_triggers.py || \
        echo "Warning: Tokenizer pre-download failed, will download at runtime if internet is available"; \
    else \
        echo "Info: PREDOWNLOAD_TOKENIZER not set, skipping tokenizer pre-download"; \
    fi

# -------- Stage 3: Minimal Runtime Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install ONLY the essential runtime libraries (no build tools, no Python installation process)
RUN apt update && \
    apt install -y libexpat1 ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire Python installation and virtual environment from python-env stage
COPY --from=python-env /usr/bin/python3.13 /usr/bin/python3.13
COPY --from=python-env /usr/lib/python3.13 /usr/lib/python3.13
COPY --from=python-env /workspace/.venv /workspace/.venv

# Copy Python shared libraries using runtime detection
RUN --mount=from=python-env,source=/usr,target=/mnt/usr \
    find /mnt/usr -name "libpython3.13.so*" -exec cp -P {} /usr/lib/ \;

# Create python3 symlink
RUN ln -s /usr/bin/python3.13 /usr/bin/python3 && \
    ln -s /usr/bin/python3.13 /usr/bin/python

# Set up environment
ENV PATH="/workspace/.venv/bin:$PATH"
WORKDIR /workspace

# Set environment variables needed for Text splitter
RUN mkdir -p /tmp-data
RUN chmod 777 -R /tmp-data
RUN chown 1000:1000 -R /tmp-data

# Set MODEL_PREDOWNLOAD_PATH and HF_HOME for runtime
ARG MODEL_PREDOWNLOAD_PATH=/workspace/models
ENV MODEL_PREDOWNLOAD_PATH=${MODEL_PREDOWNLOAD_PATH}
ENV HF_HOME=${MODEL_PREDOWNLOAD_PATH}

# Copy pre-downloaded models if they exist, otherwise create empty directory
RUN --mount=from=python-env,source=/workspace/models,target=/mnt/models \
    mkdir -p ${MODEL_PREDOWNLOAD_PATH} && \
    if [ -n "$(ls -A /mnt/models 2>/dev/null)" ]; then \
        echo "Copying pre-downloaded models..." && \
        cp -r /mnt/models/* ${MODEL_PREDOWNLOAD_PATH}/; \
    else \
        echo "No pre-downloaded models found, directory created for runtime downloads"; \
    fi

WORKDIR /workspace

ENTRYPOINT ["uvicorn", "nvidia_rag.ingestor_server.server:app"]

ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=jammy-20251013
ARG DOWNLOAD_LEGAL_COMPLIANCE=false

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

ARG DOWNLOAD_LEGAL_COMPLIANCE=false

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Snapshot packages BEFORE installing (to track only what we add) - only if legal compliance enabled
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ]; then \
        apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-before.txt; \
    fi

# Install required ubuntu packages for setting up python 3.13
# Update all packages to address security vulnerabilities including CVE-2025-68973 (gpgv)
RUN apt update && \
    apt upgrade -y && \
    apt install -y curl software-properties-common libgl1 libglib2.0-0 libmagic1 file build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt upgrade -y && apt install -y python3.13 python3.13-dev && \
    apt-get clean

# Download ONLY sources for packages WE installed (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ]; then \
        sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list && \
        find /etc/apt/sources.list.d/ -type f -name "*.list" -exec sed -i 's/# deb-src/deb-src/g' {} \; 2>/dev/null || true && \
        apt update && mkdir -p /legal/source && \
        cd /legal/source && \
        apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-after.txt && \
        comm -13 /tmp/packages-before.txt /tmp/packages-after.txt > /legal/source/packages-manifest.txt && \
        echo "Downloading sources for $(wc -l < /legal/source/packages-manifest.txt) newly installed packages (out of $(wc -l < /tmp/packages-after.txt) total)" && \
        cat /legal/source/packages-manifest.txt | \
        xargs -n 1 bash -c 'apt-get source --download-only "$0" 2>&1 || echo "Warning: Source not available for $0"' > /legal/source/download-log.txt 2>&1 && \
        echo "APT source download complete: $(du -sh /legal/source | cut -f1)"; \
    else \
        echo "Legal compliance download skipped (DOWNLOAD_LEGAL_COMPLIANCE=false)"; \
    fi

WORKDIR /workspace

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

RUN rm -rf /var/lib/apt/lists/*

# Download MPL-licensed CA Bundle from curl.se
RUN curl -fsSL https://curl.se/ca/cacert.pem -o /etc/ssl/certs/cacert.pem

WORKDIR /workspace

COPY --from=builder /build/pyproject.toml /workspace/
COPY --from=builder /build/uv.lock /workspace/

RUN uv sync --locked --no-install-project --no-dev --extra ingest --extra elasticsearch

COPY --from=builder /build/dist/*.whl /workspace/

# Find the exact wheel name with version and store it in a variable
RUN WHEEL_NAME=$(ls /workspace/nvidia_rag-*.whl) && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME"

# Pre-download default tokenizer to MODEL_PREDOWNLOAD_PATH
ARG MODEL_PREDOWNLOAD_PATH=/workspace/models
ENV MODEL_PREDOWNLOAD_PATH=${MODEL_PREDOWNLOAD_PATH}
ENV HF_HOME=${MODEL_PREDOWNLOAD_PATH}

# Copy post build triggers script and license extraction script
COPY ./src/nvidia_rag/ingestor_server/docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py
COPY ./scripts/extract_python_licenses.py /workspace/docker/extract_python_licenses.py

RUN --mount=type=cache,target=/root/.cache/pip \
    /workspace/.venv/bin/python /workspace/docker/post_build_triggers.py || \
    echo "Warning: Tokenizer pre-download failed, will download at runtime if internet is available"

# Extract Python package licenses (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ]; then \
        /workspace/.venv/bin/python /workspace/docker/extract_python_licenses.py && \
        echo "Python license extraction complete: $(du -sh /legal/python-licenses | cut -f1)"; \
    fi

# Compress legal directory to save space (conditional on DOWNLOAD_LEGAL_COMPLIANCE)
RUN if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ] && [ -d /legal ]; then \
        cd / && \
        echo "Compressing legal directory..." && \
        tar -czf /legal.tar.gz legal/ && \
        ORIGINAL_SIZE=$(du -sh /legal | cut -f1) && \
        COMPRESSED_SIZE=$(du -sh /legal.tar.gz | cut -f1) && \
        rm -rf /legal && \
        echo "Legal compliance archive created: $COMPRESSED_SIZE (original: $ORIGINAL_SIZE)"; \
    fi

# -------- Stage 3: Minimal Runtime Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS runtime

ARG DOWNLOAD_LEGAL_COMPLIANCE=false

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install ONLY the essential runtime libraries (no build tools, no Python installation process)
# Update all packages to address CVE-2025-68973 (gpgv) and other security vulnerabilities
RUN apt update && \
    apt upgrade -y && \
    apt install -y libexpat1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy MPL-licensed CA Bundle from python-env stage
COPY --from=python-env /etc/ssl/certs/cacert.pem /etc/ssl/certs/cacert.pem

# Configure system and applications to use the MPL-licensed CA bundle
ENV SSL_CERT_FILE=/etc/ssl/certs/cacert.pem
ENV CURL_CA_BUNDLE=/etc/ssl/certs/cacert.pem

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

# Set MODEL_PREDOWNLOAD_PATH and HF_HOME
ARG MODEL_PREDOWNLOAD_PATH=/workspace/models
ENV MODEL_PREDOWNLOAD_PATH=${MODEL_PREDOWNLOAD_PATH}
ENV HF_HOME=${MODEL_PREDOWNLOAD_PATH}

# Copy pre-downloaded tokenizer from python-env stage
COPY --from=python-env ${MODEL_PREDOWNLOAD_PATH} ${MODEL_PREDOWNLOAD_PATH}

# Copy legal compliance archive from python-env stage (conditional)
RUN --mount=from=python-env,source=/,target=/mnt/python-env \
    if [ "$DOWNLOAD_LEGAL_COMPLIANCE" = "true" ] && [ -f /mnt/python-env/legal.tar.gz ]; then \
        mkdir -p /legal && \
        cp /mnt/python-env/legal.tar.gz /legal/ && \
        echo "Legal compliance archive available at: /legal/legal.tar.gz" && \
        echo "Extract with: tar -xzf /legal/legal.tar.gz -C /"; \
    else \
        echo "No legal compliance archive (DOWNLOAD_LEGAL_COMPLIANCE=$DOWNLOAD_LEGAL_COMPLIANCE)"; \
    fi

WORKDIR /workspace

ENTRYPOINT ["uvicorn", "nvidia_rag.ingestor_server.server:app"]

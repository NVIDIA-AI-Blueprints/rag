ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=jammy-20251013

# -------- Stage 1: Build Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.7.4 /uv /uvx /bin/

WORKDIR /build

COPY uv.lock pyproject.toml README.md LICENSE ./
COPY ./src ./src

RUN uv build

# -------- Stage 2: Python Environment Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS python-env

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Snapshot packages BEFORE installing (to track only what we add)
RUN apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-before.txt

# Install required ubuntu packages for setting up python 3.13
RUN apt update && \
    apt install -y curl software-properties-common libgl1 libglib2.0-0 libmagic1 file build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.13 python3.13-dev && \
    apt-get clean

# Download ONLY sources for packages WE installed (not base container packages)
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list && \
    find /etc/apt/sources.list.d/ -type f -name "*.list" -exec sed -i 's/# deb-src/deb-src/g' {} \; 2>/dev/null || true && \
    apt update && mkdir -p /legal/source

WORKDIR /legal/source
RUN apt list --installed 2>/dev/null | tail -n +2 | sed 's|/.*||' | sort -u > /tmp/packages-after.txt && \
    comm -13 /tmp/packages-before.txt /tmp/packages-after.txt > /legal/source/packages-manifest.txt && \
    echo "Downloading sources for $(wc -l < /legal/source/packages-manifest.txt) newly installed packages (out of $(wc -l < /tmp/packages-after.txt) total)" && \
    cat /legal/source/packages-manifest.txt | \
    xargs -n 1 bash -c 'apt-get source --download-only "$0" 2>&1 || echo "Warning: Source not available for $0"' > /legal/source/download-log.txt 2>&1 && \
    echo "APT source download complete: $(du -sh /legal/source | cut -f1)"

WORKDIR /workspace

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

RUN rm -rf /var/lib/apt/lists/*

# Download MPL-licensed CA Bundle from curl.se
RUN curl -fsSL https://curl.se/ca/cacert.pem -o /etc/ssl/certs/cacert.pem

WORKDIR /workspace

COPY --from=builder /build/pyproject.toml /workspace/
COPY --from=builder /build/uv.lock /workspace/

RUN uv sync --locked --no-install-project --no-dev --extra rag --extra elasticsearch

COPY --from=builder /build/dist/*.whl /workspace/

# Find the exact wheel name with version and store it in a variable
RUN WHEEL_NAME=$(ls /workspace/nvidia_rag-*.whl) && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME"

# Copy license extraction script
COPY ./scripts/extract_python_licenses.py /workspace/docker/extract_python_licenses.py

# Extract Python package licenses in python-env stage (only re-runs when dependencies change)
RUN /workspace/.venv/bin/python /workspace/docker/extract_python_licenses.py && \
    echo "Python license extraction complete: $(du -sh /legal/python-licenses | cut -f1)"

# -------- Stage 3: Minimal Runtime Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install ONLY the essential runtime libraries (no build tools, no Python installation process)
RUN apt update && \
    apt install -y libexpat1 gpgv=2.2.27-3ubuntu2.5 && \
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
RUN mkdir /tmp-data/;
RUN chmod 777 -R /tmp-data
RUN chown 1000:1000 -R /tmp-data
ENV HF_HOME=/tmp-data

# Copy legal compliance artifacts from python-env stage (APT sources + Python licenses)
COPY --from=python-env /legal /legal

WORKDIR /workspace/

ENTRYPOINT ["uvicorn", "nvidia_rag.rag_server.server:app"]
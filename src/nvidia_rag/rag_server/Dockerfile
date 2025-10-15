ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=jammy-20250415.1

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

RUN uv sync --locked --no-install-project --no-dev --extra rag

COPY --from=builder /build/dist/*.whl /workspace/

# Find the exact wheel name with version and store it in a variable
RUN WHEEL_NAME=$(ls /workspace/nvidia_rag-*.whl) && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME"

# -------- Stage 3: Minimal Runtime Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install ONLY the essential runtime libraries (no build tools, no Python installation process)
RUN apt update && \
    apt install -y libexpat1 && \
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
RUN mkdir /tmp-data/;
RUN chmod 777 -R /tmp-data
RUN chown 1000:1000 -R /tmp-data
ENV HF_HOME=/tmp-data

WORKDIR /workspace/

ENTRYPOINT ["uvicorn", "nvidia_rag.rag_server.server:app"]
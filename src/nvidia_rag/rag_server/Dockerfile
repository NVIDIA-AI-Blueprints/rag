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

# -------- Stage 2: Runtime Stage --------
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.13
RUN apt update && \
    apt install -y curl software-properties-common libgl1 libglib2.0-0 libmagic1 file build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.13 python3.13-dev && \
    apt-get clean

# Install uv https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.7.4 /uv /uvx /bin/

RUN rm -rf /var/lib/apt/lists/*

# Uninstall build packages
RUN apt autoremove -y curl software-properties-common

# Download the sources of apt packages within the container for standard legal compliance
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
RUN apt update
# xz-utils is needed to pull the source and unpack them correctly
RUN apt install xz-utils -y
RUN mkdir -p /legal/source
WORKDIR /legal/source
# Read installed packages, strip all but the package names, pipe to 'apt source' to download respective packages
RUN apt list --installed | sed 's|/.*||' | sort -u | \
    xargs -n 1 bash -c 'apt source --download-only "$0" || true'
# The source is saved in directories as well as tarballs in the current dir
RUN rm xz-utils*
COPY LICENSE-3rd-party.txt /legal/

WORKDIR /workspace

COPY --from=builder /build/pyproject.toml /workspace/
COPY --from=builder /build/uv.lock /workspace/

RUN uv sync --locked --no-install-project --no-dev --extra rag

COPY --from=builder /build/dist/*.whl /workspace/

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

# Find the exact wheel name with version and store it in a variable
RUN WHEEL_NAME=$(ls /workspace/nvidia_rag-*.whl) && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME"

# Remove executables and wheel
RUN rm /bin/uv /bin/uvx /workspace/nvidia_rag-*.whl  /workspace/pyproject.toml  /workspace/uv.lock
RUN apt autoremove -y build-essential

# Set environment variables needed for Text splitter
RUN mkdir /tmp-data/;
RUN chmod 777 -R /tmp-data
RUN chown 1000:1000 -R /tmp-data
ENV HF_HOME=/tmp-data

WORKDIR /workspace/

ENTRYPOINT ["uvicorn", "nvidia_rag.rag_server.server:app"]
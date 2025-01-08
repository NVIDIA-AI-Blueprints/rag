ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.10
RUN apt update && \
    apt install -y openssl libgl1 linux-libc-dev libksba8 curl software-properties-common build-essential libssl-dev libffi-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.10 python3.10-dev python3.10-distutils

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN rm -rf /var/lib/apt/lists/*

# Install all custom python dependencies
RUN mkdir /app
COPY ./frontend/requirements.txt /app
RUN apt-get update; \
    apt-get upgrade -y; \
    pip3 --no-cache-dir install -r /app/requirements.txt;

# Uninstall build packages
RUN apt autoremove -y openssl libgl1 linux-libc-dev libksba8 curl software-properties-common build-essential libssl-dev libffi-dev python3.10-dev python3.10-distutils

RUN apt-get clean

COPY ./frontend /app/frontend

WORKDIR /app
ENTRYPOINT ["python3.10", "-m", "frontend"]

FROM 192.168.1.165:30002/kt-moa/hardcap-api-server:latest

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    software-properties-common \
    autoconf \
    automake \
    libtool \
    pkg-config \
    ca-certificates \
    wget \
    git \
    curl \
    libjpeg-dev \
    libpng-dev \
    language-pack-en \
    locales \
    locales-all \
    python3 \
    python3-py \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-pytest \
    python3-setuptools \
    libprotobuf-dev \
    protobuf-compiler \
    zlib1g-dev \
    swig \
    vim \
    gdb \
    valgrind \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    unzip \
    libgl1-mesa-glx
RUN apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

COPY requirements.txt /home/requirements.txt

RUN cd /usr/local/bin && \
    pip3 install --upgrade pip setuptools wheel && \
    cd /home/ && \
    python3 --version && \
    pip3 install -r requirements.txt

RUN cd /root/ && \
    mkdir -p /root/bentoml

RUN curl -fsSL get.docker.com -o get-docker.sh
RUN sh get-docker.sh


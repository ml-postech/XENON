FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility,display

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Acquire::GzipIndexes \"false\"; Acquire::CompressionTypes::Order:: \"gz\";" > /etc/apt/apt.conf.d/docker-gzip-indexes

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    less \
    emacs \
    tmux \
    bash-completion \
    command-not-found \
    software-properties-common \
    curl \
    wget \
    unzip \
    htop \
    git \
    build-essential \
    pkg-config \
    xdg-user-dirs \
    ca-certificates \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    freeglut3-dev \
    mesa-utils \
    mesa-utils-extra \
    vulkan-tools \
    libvulkan-dev \
    libglfw3-dev \
    libegl1-mesa-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libsm6 \
    libxext6 \
    libvulkan1 \
    vulkan-tools \
    libglvnd-dev \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nano \
    vim \
    wget \
    bzip2 \
    gzip \
    zip \
    unzip \
    zstd \
    gcc \
    git \
    jq \
    make \
    curl \
    cmake \
    clang \
    openjdk-8-jdk \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    xvfb \
    xorg-dev \
    dbus-x11 \
    x11-utils \
    x11-xserver-utils \
    libdbus-c++-1-0v5 \
    dmz-cursor-theme \
    numlockx \
    xcursor-themes \
    libglu1-mesa-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir -p /opt/conda && mkdir -p /app && mkdir -p /app

WORKDIR /app/

# Miniconda install
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-0-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -u -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:/app:$PATH"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update -n base -c defaults conda -y && \
    conda clean -afy

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/app:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"



# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install torch
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir transformers==4.57.6 sentence-transformers==3.4.1 accelerate qwen-vl-utils[decord]

RUN pip install --no-cache-dir \
    numpy==1.23.5 Pillow scipy opencv-python pynput termcolor regex retrying tqdm typing_extensions autoawq jinja2 \
    requests ipython typing lxml psutil Pyro4 getch coloredlogs dill daemoniker xmltodict==0.12.0 inflection imagehash flaky pyglet

RUN pip install --no-cache-dir \
    setuptools==65.5.1 wheel==0.38.0 x_transformers==0.27.1 dm-tree==0.1.9 \
    wandb timm sentencepiece attrdict einops hydra-core shortuuid rich thefuzz ruff fastapi pyvirtualdisplay openai av uvicorn attrs
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir gym3 gym==0.23.0


# docker build --tag xenon:0.1 .


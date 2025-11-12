# syntax=docker/dockerfile:1

# GPU-enabled base image with CUDA 12.1 and build tools for compiling extensions
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1

# System deps: Python, build essentials, and libraries for OpenCV/Open3D/nvdiffrast
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python-is-python3 \
    git build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build COLMAP from source with CUDA support
# Install COLMAP build dependencies (per official docs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev \
    libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and compile COLMAP v3.13.0 enabling CUDA; target common Ampere/Lovelace architectures
RUN git clone --branch 3.13.0 --depth 1 --recursive https://github.com/colmap/colmap.git /opt/colmap \
 && cd /opt/colmap \
 && git submodule update --init --recursive \
 && mkdir -p build \
 && cd build \
 && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
 && ninja \
 && ninja install \
 && ldconfig

# Quick sanity check to ensure COLMAP is installed and linked with CUDA
RUN colmap -h >/dev/null || true

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

# Copy requirements for dependency install (uses pip VCS and CUDA-aware libs)
COPY requirements.txt ext_requirements.txt /workspace/

# Install base dependencies first (no VCS, no CUDA extensions)
RUN python3 -m pip install --no-cache-dir -r /workspace/ext_requirements.txt

# Install VCS/CUDA-aware packages with access to the existing torch
RUN python3 -m pip install --no-cache-dir --no-build-isolation -r /workspace/requirements.txt

# Install gsplat
RUN python3 -m pip install --no-build-isolation "git+https://github.com/dendenxu/gsplat.git" 

# Copy project sources
COPY . /workspace

# Build local CUDA/C++ extensions (e.g., simple-knn)
RUN python3 -m pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn

# Default to interactive shell; bind-mount and run training as needed
CMD ["bash"]
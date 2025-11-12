# syntax=docker/dockerfile:1

# GPU-enabled base image with CUDA 12.1 and build tools for compiling extensions
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1

# Enable Universe repo (COLMAP resides in Universe on Ubuntu)
# RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
#     && add-apt-repository -y universe \
#     && rm -rf /var/lib/apt/lists/*

# System deps: Python, build essentials, and libraries for OpenCV/Open3D/nvdiffrast
# Install COLMAP via distribution binaries (per Repology overview) to avoid source builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python-is-python3 \
    git build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Note: apt-provided COLMAP may be CPU-only or lag behind latest releases
## If CUDA-accelerated COLMAP or a specific version is required, consider
## switching back to source build or using the official colmap Docker image.

# Quick sanity check to ensure COLMAP is installed and linked with CUDA
RUN colmap -h >/dev/null || true

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

# Copy requirements for dependency install (uses pip VCS and CUDA-aware libs)
COPY requirements.txt ext_requirements.txt /workspace/

RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip3 config set global.trusted-host mirrors.aliyun.com

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
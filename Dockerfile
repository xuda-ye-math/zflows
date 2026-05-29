# zflows on CUDA 13 (latest packages, default Python)
#
# build the container
#   docker build -t xudayemath/zflows:cu13 .
# push to docker hub
#   docker push xudayemath/zflows:cu13
# interactive shell
#   docker run --rm -it --gpus all xudayemath/zflows:cu13
# mount current dir and work in it
#   docker run --rm -it --gpus all -v "$PWD:/workspace" xudayemath/zflows:cu13
# python REPL
#   docker run --rm -it --gpus all xudayemath/zflows:cu13 python
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

# Non-interactive apt, unbuffered Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Python (Ubuntu 24.04 default: python3.12) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Virtual env named "torch", matching ~/.envs/torch ---
ENV VIRTUAL_ENV=/opt/torch
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip setuptools wheel

# --- PyTorch built against CUDA 13 (pulls the bundled nvidia-*-cu13 wheels) ---
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch \
        torchvision

# --- Core scientific stack ---
RUN pip install \
        numpy \
        scipy \
        scikit-learn \
        pandas \
        matplotlib \
        h5py \
        tqdm

# --- zflows (from PyPI) ---
# torch is already satisfied above, so this only adds zflows itself.
RUN pip install zflows

WORKDIR /workspace

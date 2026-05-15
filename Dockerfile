FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# system deps
RUN apt-get update && apt-get install -y \
    tmux \
    git \
    build-essential \
    g++-11 \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

WORKDIR /app

COPY requirements.txt .
COPY patch_dependencies.py .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# PyG wheels
RUN pip install --no-build-isolation \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# extra deps for xLSTM
RUN pip install \
    mlstm_kernels \
    xlstm

# patch installed dependencies
RUN python /app/patch_dependencies.py

# syntax checks
RUN python -m py_compile \
    /opt/conda/lib/python3.11/site-packages/xlstm/xlstm_large/model.py

WORKDIR /workspace
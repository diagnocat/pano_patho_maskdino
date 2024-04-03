FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

ENV TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0 8.6'
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    \
    apt-get install -y git tmux build-essential libgl1-mesa-glx libglib2.0-0 && \
    \
    apt-get clean && \
    apt-get autoremove -yqq && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@898507047cf441a1e4be7a729270961c401c4354'

RUN conda install shapely --channel conda-forge

COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

WORKDIR /workspace/multi-scale-deformable-attention
COPY src/maskdino/multi-scale-deformable-attention .
RUN make install

WORKDIR /workspace

ENV SHELL=/usr/bin/bash

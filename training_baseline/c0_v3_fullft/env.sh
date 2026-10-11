#!/usr/bin/env bash
# source 本文件只设置本次训练使用的路径；不更改系统 Python。
export PI05_RUNTIME_DIR="${PI05_RUNTIME_DIR:-/home/zhiwei/pi05-runtime}"
export TMPDIR="$PI05_RUNTIME_DIR/tmp"
export UV_CACHE_DIR="$PI05_RUNTIME_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PI05_RUNTIME_DIR/python"
export UV_PROJECT_ENVIRONMENT="$PI05_RUNTIME_DIR/venv"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
# PyPI 与阿里云内网镜像绕过本机代理；大文件优先复用已校验的 wheel 缓存。
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}pypi.org,files.pythonhosted.org,mirrors.cloud.aliyuncs.com"
export no_proxy="$NO_PROXY"
export OPENPI_DATA_HOME="$PI05_RUNTIME_DIR/openpi-cache"
export HF_HOME="$PI05_RUNTIME_DIR/huggingface"
export CUDA_CACHE_PATH="$PI05_RUNTIME_DIR/cuda-cache"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
# 本机已有可用 Python 3.11 时复用解释器，依赖仍安装到独立 venv。
if [[ -z "${PI05_PYTHON:-}" && -x /mnt/data/public_tools/miniconda3/envs/rrs_p1/bin/python3.11 ]]; then
    export PI05_PYTHON=/mnt/data/public_tools/miniconda3/envs/rrs_p1/bin/python3.11
fi
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$OPENPI_DATA_HOME" "$HF_HOME" "$CUDA_CACHE_PATH"

#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}

export MAX_JOBS=32

#! activate ur conda env

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.4.6.post3" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && uv pip install torch-memory-saver --no-cache-dir
fi
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" "torchdata<=0.11.0"

echo "2. install basic packages"
# Pin transformers>=4.51.0,<5.0.0: need >=4.51 for Qwen3, <5.0 for vllm 0.8.5 (all_special_tokens_extended removed in 5.0)
# Tested with 4.57.6
# Max versions pinned to avoid future breaking changes (2025-01-26)
pip install "transformers[hf_xet]>=4.51.0,<5.0.0" "accelerate<=1.12.0" "datasets<=4.5.0" "peft<=0.18.1" "hf-transfer<=0.1.9" \
    "numpy<2.0.0" "pyarrow>=15.0.0,<=22.0.0" "pandas<=2.3.3" \
    "ray[default]<=2.53.0" "codetiming<=1.4.0" "hydra-core<=1.3.2" "pylatexenc<=2.10" "qwen-vl-utils<=0.0.14" "wandb<=0.23.1" "dill<=0.4.0" "pybind11<=3.0.1" "liger-kernel<=0.6.4" "mathruler<=0.1.0" \
    "pytest<=9.0.2" "py-spy<=0.4.1" "pyext<=0.7" "pre-commit<=4.5.1" "ruff<=0.14.14" "tensorboard<=2.20.0" "math_verify<=0.9.0"

pip install "nvidia-ml-py>=12.560.30,<=13.580.82" "fastapi[standard]>=0.115.0,<=0.128.0" "optree>=0.13.0,<=0.18.0" "pydantic>=2.9,<=2.12.5" "grpcio>=1.62.1,<=1.76.0"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi


echo "5. May need to fix opencv"
pip install "opencv-python<=4.13.0.90"
pip install "opencv-fixer<=0.2.5" && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages"

# pip install sglang[all]==0.4.9.post6
# pip uninstall flash-attn
# pip install flash-attn --no-build-isolation
#these work
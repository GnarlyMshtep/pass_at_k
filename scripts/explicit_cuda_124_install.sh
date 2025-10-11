#!/bin/bash
USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-0}
export MAX_JOBS=32

# Explicitly set CUDA version
CUDA_VERSION="12.4"
TORCH_CUDA="cu124"  # for torch index-url format

echo "Using CUDA version: ${CUDA_VERSION} (${TORCH_CUDA})"

#! activate ur conda env
echo "1. install inference frameworks and pytorch they need"

# Install PyTorch with explicit CUDA version
echo "Installing PyTorch 2.6.0 with CUDA ${CUDA_VERSION}"
pip install --no-cache-dir \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# Install other torch-related packages
pip install --no-cache-dir "tensordict==0.6.2" torchdata

# Install vLLM with explicit CUDA support
echo "Installing vLLM 0.8.5.post1 with CUDA ${CUDA_VERSION}"
pip install --no-cache-dir "vllm==0.8.5.post1"

if [ $USE_SGLANG -eq 1 ]; then
    echo "Installing SGLang with CUDA ${CUDA_VERSION}"
    pip install "sglang[all]==0.4.6.post3" --no-cache-dir \
        --find-links https://flashinfer.ai/whl/${TORCH_CUDA}/torch2.6/flashinfer-python
    pip install torch-memory-saver --no-cache-dir
fi

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff tensorboard math_verify
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashAttention and FlashInfer with explicit CUDA ${CUDA_VERSION}"

# Install flash-attn-2.7.4.post1 for CUDA 12 + torch 2.6 (cxx11abi=False)
FLASH_ATTN_WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
echo "Downloading ${FLASH_ATTN_WHEEL}"
wget -nv "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHEEL}" && \
    pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"

# Install flashinfer-0.2.2.post1 for CUDA 12.4 + torch 2.6 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
FLASHINFER_WHEEL="flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
echo "Downloading ${FLASHINFER_WHEEL}"
wget -nv "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/${FLASHINFER_WHEEL}" && \
    pip install --no-cache-dir "${FLASHINFER_WHEEL}"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron with CUDA ${CUDA_VERSION}"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    
    # Set CUDA_HOME if not set (TransformerEngine needs this)
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
        echo "Set CUDA_HOME=${CUDA_HOME}"
    fi
    
    # Install TransformerEngine with explicit CUDA support
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_USERBUFFERS=1 \
    pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
    
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi

echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package for CUDA 12 (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages with CUDA ${CUDA_VERSION}"
echo ""
echo "Verification commands:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo "  python -c 'import torch; print(f\"CUDA version: {torch.version.cuda}\")'"
echo "  python -c 'import torch; print(f\"GPU count: {torch.cuda.device_count()}\")'"
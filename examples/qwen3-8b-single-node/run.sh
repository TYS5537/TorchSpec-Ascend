#!/bin/bash
# Train with SGLang/vLLM async inference (multi-GPU version)
# Supports both NVIDIA CUDA and Ascend NPU.
#
# Device allocation:
#   CUDA (default: 4 GPUs total):
#     - 2 GPUs for inference (SglEngine, TP=2)
#     - 2 GPUs for training (FSDP/DP: model sharded across 2 GPUs)
#   Ascend NPU (default: 3 NPUs total):
#     - 1 NPU for inference (HFEngine — SGLang does not yet support Ascend NPU)
#     - 2 NPUs for training (FSDP/DP: model sharded across 2 NPUs)
#
# Usage:
#   ./examples/qwen3-8b-single-node/run.sh [CONFIG_FILE] [EXTRA_ARGS...]
#
# Examples:
#   # CUDA — run with default multi-GPU config
#   ./examples/qwen3-8b-single-node/run.sh
#
#   # CUDA — run with custom config
#   ./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml
#
#   # CUDA — run with extra args
#   ./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml training.num_train_steps=10
#
#   # Ascend NPU — auto-detected via ASCEND_RT_VISIBLE_DEVICES
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2 ./examples/qwen3-8b-single-node/run.sh
#
#   # Ascend NPU — explicit device type
#   TORCHSPEC_DEVICE_TYPE=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2 \
#     ./examples/qwen3-8b-single-node/run.sh configs/hf_qwen3_8b.yaml

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHSPEC_LOG_LEVEL=INFO

# ── Device detection ──────────────────────────────────────────────────────────
# TORCHSPEC_DEVICE_TYPE takes precedence; otherwise auto-detect from env vars.
if [[ -z "${TORCHSPEC_DEVICE_TYPE:-}" ]]; then
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        export TORCHSPEC_DEVICE_TYPE="npu"
    else
        export TORCHSPEC_DEVICE_TYPE="cuda"
    fi
fi

if [[ "$TORCHSPEC_DEVICE_TYPE" == "npu" ]]; then
    # ── Ascend NPU ────────────────────────────────────────────────────────────
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2}"
    # Prevent Ray from overriding device visibility on NPU workers.
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    # HCCL is the Ascend equivalent of NCCL; set the network interface here.
    export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
    # TorchInductor is not used on NPU; omit TORCHINDUCTOR_CACHE_DIR.
    IFS=',' read -ra DEVICE_ARRAY <<< "$ASCEND_RT_VISIBLE_DEVICES"
    TOTAL_DEVICES=${#DEVICE_ARRAY[@]}
    TRAIN_DEVICES=2
    INFERENCE_DEVICES=1
    VISIBLE_DEVICES_MSG="ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
else
    # ── NVIDIA CUDA ───────────────────────────────────────────────────────────
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
    # Prevent Ray from overriding CUDA_VISIBLE_DEVICES on CUDA workers.
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    IFS=',' read -ra DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    TOTAL_DEVICES=${#DEVICE_ARRAY[@]}
    TRAIN_DEVICES=2
    INFERENCE_DEVICES=2
    VISIBLE_DEVICES_MSG="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

CONFIG_FILE="${1:-$ROOT_DIR/configs/sglang_qwen3_8b.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/sglang_qwen3_8b.yaml"
fi

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Train with async inference"
echo "=============================================="
echo "Config:          $CONFIG_FILE (nested format)"
echo "Device:          $TORCHSPEC_DEVICE_TYPE"
echo "Total devices:   $TOTAL_DEVICES ($VISIBLE_DEVICES_MSG)"
echo "  - Training:    $TRAIN_DEVICES (FSDP/DP - model sharded)"
echo "  - Inference:   $INFERENCE_DEVICES"
echo "Local IP:        $LOCAL_IP"
echo "Extra args:      $*"
echo "=============================================="

if [[ "$TORCHSPEC_DEVICE_TYPE" == "npu" ]]; then
    # NPU: HFEngine (SGLang lacks Ascend support); sdpa replaces flex_attention
    # (Triton-only); hccl replaces nccl for collective communication.
    python3 -m torchspec.train_entry \
        --config "$CONFIG_FILE" \
        training.training_num_gpus_per_node="$TRAIN_DEVICES" \
        training.attention_backend=sdpa \
        training.distributed_backend=hccl \
        inference.inference_engine_type=hf \
        inference.inference_num_gpus="$INFERENCE_DEVICES" \
        inference.inference_num_gpus_per_engine=1 \
        inference.inference_num_gpus_per_node="$TOTAL_DEVICES" \
        "$@"
else
    # TODO: unify tp_size config across sglang/vllm backends
    python3 -m torchspec.train_entry \
        --config "$CONFIG_FILE" \
        training.training_num_gpus_per_node="$TRAIN_DEVICES" \
        inference.inference_num_gpus="$INFERENCE_DEVICES" \
        inference.inference_num_gpus_per_engine=2 \
        inference.inference_num_gpus_per_node="$TOTAL_DEVICES" \
        inference.sglang.tp_size=2 \
        "$@"
fi

echo "=============================================="
echo "Training completed!"
echo "=============================================="

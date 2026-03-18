#!/usr/bin/env bash
set -e

export HF_HOME=/root/autodl-tmp/hf_models
export VLLM_USE_V1=1

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=8000

mkdir -p logs

nohup vllm serve "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --generation-config vllm \
  > logs/vllm_qwen_7b.out 2>&1 &
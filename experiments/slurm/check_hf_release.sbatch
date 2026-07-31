#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -J nx-hfcheck

source "experiments/slurm/_common.sh"

ADAPTER_PACKAGE="release/huggingface/ninaxander-raven7b-tulu69-adapter"
PYTHIA_TO_RWKV_PACKAGE="release/huggingface/ninaxander-tulu69-to-raven7b-best"
RWKV_TO_PYTHIA_PACKAGE="release/huggingface/ninaxander-raven7b-to-tulu69-best"

"$PYTHON" -u "$ADAPTER_PACKAGE/validate_package.py" \
  --package "$ADAPTER_PACKAGE" \
  --source-checkpoint "$FINAL_ADAPTER"

CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u \
  tools/check_cross_family_chimera.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/cross_family_runtime_smoke.json"

for direction in rwkv-to-pythia pythia-to-rwkv; do
  CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u \
    "$ADAPTER_PACKAGE/example_generate.py" \
    --adapter "$ADAPTER_PACKAGE" \
    --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tokenizer "$TOKENIZER" \
    --direction "$direction" --switch 4 --max-new-tokens 1 \
    --device-a cuda:0 --device-b cuda:1
done

for package in "$PYTHIA_TO_RWKV_PACKAGE" "$RWKV_TO_PYTHIA_PACKAGE"; do
  CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u \
    "$package/validate_package.py" \
    --package "$package" \
    --source-checkpoint "$FINAL_ADAPTER" \
    --gpu-smoke --device-a cuda:0 --device-b cuda:1
done

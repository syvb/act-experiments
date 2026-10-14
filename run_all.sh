#!/usr/bin/env bash
# Full GPU run: means + patching for every dense Qwen3 base model, then plots. Run from the repo root on the pod:
#   nohup ./run_all.sh > run.log 2>&1 &
# and copy out/ off the pod before terminating it (the container disk is deleted with the pod).
# If ~/.wandb_key exists, each model's result curves are logged to W&B as soon as that model is done.
set -euo pipefail

TOKENS=${TOKENS:-10e6}
SAMPLES=${SAMPLES:-3000}
MODELS=${MODELS:-"0.6B 1.7B 4B 8B 14B"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "== $(date -u +%T) $*"; }
if [ -f ~/.wandb_key ]; then export WANDB_API_KEY=$(cat ~/.wandb_key); fi

log "install"
python -m pip install -q -r requirements.txt
mkdir -p data out/plots

# Download all models in the background (in order) so downloads overlap with compute.
(for size in $MODELS; do hf download Qwen/Qwen3-$size-Base > /dev/null; done) &

if [ ! -f data/fineweb_000.parquet ]; then
    log "download data"
    hf download HuggingFaceFW/fineweb sample/10BT/000_00000.parquet --repo-type dataset --local-dir data/dl
    ln -s "$PWD/data/dl/sample/10BT/000_00000.parquet" data/fineweb_000.parquet
fi

log "test"
python test_patch.py

for size in $MODELS; do
    model=Qwen/Qwen3-$size-Base
    if [ ! -f out/means_$size.pt ]; then
        log "means $size"
        python means.py --model $model --data data/fineweb_000.parquet --tokens $TOKENS --out out/means_$size.pt
    fi
    if [ ! -f out/metrics_$size.npz ]; then
        log "patch $size"
        python patch.py --model $model --data data/fineweb_000.parquet --means out/means_$size.pt \
            --samples $SAMPLES --out out/metrics_$size.npz --acts out/acts_$size.pt
        if [ -n "${WANDB_API_KEY:-}" ]; then python log_wandb.py out/metrics_$size.npz || log "wandb logging failed"; fi
    fi
done

# Check on the largest model that bf16 means match fp32 ones (see check_means_precision.py).
big=${MODELS##* }
if [ ! -f out/check_means_$big.txt ]; then
    log "precision check $big"
    for dt in bfloat16 float32; do
        python means.py --model Qwen/Qwen3-$big-Base --data data/fineweb_000.parquet --tokens 3e5 --min-count 100 \
            --dtype $dt --out out/check_means_${big}_$dt.pt
    done
    python check_means_precision.py out/check_means_${big}_bfloat16.pt out/check_means_${big}_float32.pt \
        out/metrics_$big.npz > out/check_means_$big.txt
fi

log "plot"
python plot.py out/plots $(for size in $MODELS; do echo out/metrics_$size.npz; done) | tee out/plots/summary.txt
log "done"

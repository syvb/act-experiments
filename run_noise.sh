#!/usr/bin/env bash
# Noise pass on the GPU: mean and resample ablation plus noise added to the clean residual (see patch.py), for one
# model. With the default --samples 3000 and seed, the samples and donors are the same as in the main run. From the
# repo root on the pod:
#   nohup ./run_noise.sh > noise.log 2>&1 &
# The means come from the main run (on Hugging Face). Copy out/ off the pod before terminating it.
set -euo pipefail

SIZE=${SIZE:-14B}
SCALES=${SCALES:-"0.25 1 4 16"}
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "== $(date -u +%T) $*"; }
if [ -f ~/.wandb_key ]; then export WANDB_API_KEY=$(cat ~/.wandb_key); fi

log "install"
python -m pip install -q -r requirements.txt
mkdir -p data out/plots_noise
hf download Qwen/Qwen3-$SIZE-Base > /dev/null &  # in the background, while the rest downloads and the test runs
if [ ! -f data/fineweb_000.parquet ]; then
    log "download data"
    hf download HuggingFaceFW/fineweb sample/10BT/000_00000.parquet --repo-type dataset --local-dir data/dl
    ln -s "$PWD/data/dl/sample/10BT/000_00000.parquet" data/fineweb_000.parquet
fi
if [ ! -f out/means_$SIZE.pt ]; then
    log "download means"
    hf download syvb/qwen3-last-position-patching means_$SIZE.pt --repo-type dataset --local-dir out
fi

log "test"
python test_patch.py
wait

log "noise $SIZE"
python patch.py --model Qwen/Qwen3-$SIZE-Base --data data/fineweb_000.parquet --means out/means_$SIZE.pt \
    --samples 3000 --ablations mean resample --noise-scales $SCALES --out out/metrics_noise_$SIZE.npz
if [ -n "${WANDB_API_KEY:-}" ]; then
    python log_wandb.py out/metrics_noise_$SIZE.npz qwen3-self-correction-noise || log "wandb logging failed"
fi

log "plot"
python plot_noise.py out/plots_noise out/metrics_noise_$SIZE.npz > out/plots_noise/summary_$SIZE.txt
log "done"

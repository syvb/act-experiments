#!/usr/bin/env bash
# Full GPU run: means + patching for every dense Qwen3 base model, then plots. Run from the repo root on the pod:
#   nohup ./run_all.sh > run.log 2>&1 &
# and copy out/ off the pod before terminating it (the container disk is deleted with the pod).
set -euo pipefail

TOKENS=${TOKENS:-10e6}
SAMPLES=${SAMPLES:-3000}
MODELS=${MODELS:-"0.6B 1.7B 4B 8B 14B"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "== $(date -u +%T) $*"; }

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
            --samples $SAMPLES --out out/metrics_$size.npz
    fi
done

log "plot"
python plot.py out/plots $(for size in $MODELS; do echo out/metrics_$size.npz; done) | tee out/plots/summary.txt
log "done"

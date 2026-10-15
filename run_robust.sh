#!/usr/bin/env bash
# Extra pass run after run_all.sh: 1000 more samples per model that also record distances without the 5
# massive-activation dims (in Qwen3-8B, a few samples reach |x| ~ 1e4 there and dominate L2 averages).
set -euo pipefail
MODELS=${MODELS:-"0.6B 1.7B 4B 8B 14B"}
export PYTORCH_ALLOC_CONF=expandable_segments:True
for size in $MODELS; do
    echo "== $(date -u +%T) robust $size"
    python patch.py --model Qwen/Qwen3-$size-Base --data data/fineweb_000.parquet --means out/means_$size.pt \
        --samples 1000 --out out/metrics_robust_$size.npz
done
mkdir -p out/plots_robust
python plot.py out/plots_robust $(for size in $MODELS; do echo out/metrics_robust_$size.npz; done) \
    > out/plots_robust/summary.txt
echo "== $(date -u +%T) robust done"

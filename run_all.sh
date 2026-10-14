#!/usr/bin/env bash
# Full GPU run: means + patching for every dense Qwen3 base model, then plots. Run from the repo root on the pod.
set -euo pipefail

TOKENS=${TOKENS:-10e6}
SAMPLES=${SAMPLES:-5000}
MODELS=${MODELS:-"0.6B 1.7B 4B 8B 14B"}

pip install -q -r requirements.txt
mkdir -p data out
[ -f data/fineweb_000.parquet ] || hf download HuggingFaceFW/fineweb sample/10BT/000_00000.parquet \
    --repo-type dataset --local-dir data/dl && ln -sf "$PWD/data/dl/sample/10BT/000_00000.parquet" data/fineweb_000.parquet

python test_patch.py

for size in $MODELS; do
    model=Qwen/Qwen3-$size-Base
    [ -f out/means_$size.pt ] || python means.py --model $model --data data/fineweb_000.parquet --tokens $TOKENS \
        --out out/means_$size.pt
    [ -f out/metrics_$size.npz ] || python patch.py --model $model --data data/fineweb_000.parquet \
        --means out/means_$size.pt --samples $SAMPLES --out out/metrics_$size.npz
done

python plot.py out/plots $(for size in $MODELS; do echo out/metrics_$size.npz; done)

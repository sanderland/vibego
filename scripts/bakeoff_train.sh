#!/usr/bin/env bash
# Arch bake-off: train old vs nbt vs nbt-param-matched at the 6b and 10b scales on the SAME
# distilled data for the SAME step budget (fair: equal data/gradient-steps each). Intrinsic eval
# (held-out val loss + policy_eval vs a neutral judge) decides which go head-to-head.
#   STEPS=8000 DATA=distilled_1m scripts/bakeoff_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints/bakeoff
STEPS=${STEPS:-8000}
DATA=${DATA:-distilled_1m}

# name : arch   (name is stable; arch comes from the model registry)
NETS=(
  "old6b:b6c96-gpool"        # classic ResNet, 1.09M
  "nbt6b:b6c96nbt"           # nbt same b/c label, 0.80M (smaller — shows nbt efficiency)
  "nbt6b_matched:b6c112nbt"  # nbt widened to ~old6b budget, 1.07M (same budget, better block?)
  "old10b:b10c128-gpool"     # classic ResNet, 3.12M
  "nbt10b:b10c128nbt"        # nbt same b/c label, 2.21M
  "nbt10b_matched:b10c152nbt" # nbt widened to ~old10b budget, 3.07M
)

for entry in "${NETS[@]}"; do
  name="${entry%%:*}"; arch="${entry##*:}"
  out="checkpoints/bakeoff/${name}.pt"
  log="checkpoints/bakeoff/${name}.log"
  if [ -f "$out" ]; then echo "[skip] $name already trained -> $out"; continue; fi
  echo "=== [$(date +%H:%M:%S)] train $name ($arch), $STEPS steps ==="
  uv run python scripts/train.py --data "$DATA" --arch "$arch" \
    --batch-size 256 --max-steps "$STEPS" --warmup 200 \
    --eval-interval 2000 --save-interval 4000 --val-files 4 \
    --out "$out" > "$log" 2>&1
  echo "    done -> $out  ($(grep -c '^step' "$log") logged steps)"
  grep '\[eval step' "$log" | tail -1 || true
done
echo "=== ALL DONE [$(date +%H:%M:%S)] ==="

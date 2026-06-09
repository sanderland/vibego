#!/usr/bin/env bash
# Depth-vs-width study, all nbt, every net ~param-matched to old6b (~1.09M). Same data/steps as the
# main bake-off. WAITS for the main bake-off to free the GPU first (no MPS contention), then trains
# depths 7..10 (depth 6 = the main bake-off's nbt6b_matched / b6c112nbt, reused).
#   STEPS=8000 DATA=distilled_1m scripts/bakeoff_depthwidth.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints/bakeoff
STEPS=${STEPS:-8000}
DATA=${DATA:-distilled_1m}

# Wait until the main 6-net bake-off has produced all its checkpoints (so the GPU is free).
need=(old6b nbt6b nbt6b_matched old10b nbt10b nbt10b_matched)
echo "=== [$(date +%H:%M:%S)] waiting for main bake-off to finish before starting depth-width ==="
while true; do
  missing=0
  for n in "${need[@]}"; do [ -f "checkpoints/bakeoff/${n}.pt" ] || missing=1; done
  # also make sure no train.py is still running (last net mid-flight)
  if [ "$missing" = "0" ] && ! pgrep -f "scripts/train.py" >/dev/null; then break; fi
  sleep 60
done
echo "=== [$(date +%H:%M:%S)] GPU free — starting depth-width study ==="

NETS=(
  "dw7_b7c106:b7c106nbt"
  "dw8_b8c102:b8c102nbt"
  "dw9_b9c92:b9c92nbt"
  "dw10_b10c88:b10c88nbt"
)
for entry in "${NETS[@]}"; do
  name="${entry%%:*}"; arch="${entry##*:}"
  out="checkpoints/bakeoff/${name}.pt"; log="checkpoints/bakeoff/${name}.log"
  if [ -f "$out" ]; then echo "[skip] $name already trained -> $out"; continue; fi
  echo "=== [$(date +%H:%M:%S)] train $name ($arch), $STEPS steps ==="
  uv run python scripts/train.py --data "$DATA" --arch "$arch" \
    --batch-size 256 --max-steps "$STEPS" --warmup 200 \
    --eval-interval 2000 --save-interval 4000 --val-files 4 \
    --out "$out" > "$log" 2>&1
  echo "    done -> $out"; grep '\[eval step' "$log" | tail -1 || true
done
echo "=== DEPTH-WIDTH DONE [$(date +%H:%M:%S)] ==="

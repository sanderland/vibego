#!/usr/bin/env bash
# One-shot bootstrap for a RunPod (or any CUDA box): clone -> uv sync -> install KataGo +
# models -> download data -> relabel with a strong teacher (b18) -> train -> judge vs b6c96.
#
#   bash scripts/runpod_bootstrap.sh
#
# Everything is env-overridable (see CONFIG). Set SMOKE=1 for a tiny CPU end-to-end that needs
# no GPU/KataGo/network (uses nanogo as its own teacher) — used to test this script in Docker:
#   docker run --rm -i python:3.12-slim bash -c '...'   (see the repo for the exact command)
set -euo pipefail
log() { printf '\n=== %s ===\n' "$*"; }

# ---------------- CONFIG ----------------
SMOKE=${SMOKE:-0}
REPO=${REPO:-https://github.com/sanderland/nanogo.git}
BRANCH=${BRANCH:-dev}
WORKDIR=${WORKDIR:-/workspace/nanogo}
ARCH=${ARCH:-b6c96-gpool}
DATA_FROM=${DATA_FROM:-2021-06-01}
DATA_DAYS=${DATA_DAYS:-5}
RELABEL_FILES=${RELABEL_FILES:-8000}     # ~52 positions/file -> ~400k positions
TRAIN_STEPS=${TRAIN_STEPS:-60000}
# KataGo prebuilt: MATCH this to the pod's CUDA/TensorRT. See github.com/lightvector/KataGo/releases
KATAGO_URL=${KATAGO_URL:-https://github.com/lightvector/KataGo/releases/download/v1.16.4/katago-v1.16.4-trt10.6.0-cuda12.6-linux-x64.zip}
TEACHER_URL=${TEACHER_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}
B6_URL=${B6_URL:-https://katagoarchive.org/g170/neuralnets/g170-b6c96-s175395328-d26788732.bin.gz}

# ---------------- clone + env ----------------
if [ ! -f "$WORKDIR/pyproject.toml" ]; then
  log "Clone $REPO@$BRANCH -> $WORKDIR"
  git clone -b "$BRANCH" "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

if ! command -v uv >/dev/null 2>&1; then
  log "Install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

log "uv sync"
uv sync --extra dev
log "Run tests (KataGo-dependent ones skip if the submodule isn't present)"
uv run pytest -q

# ---------------- SMOKE: tiny CPU end-to-end, no GPU/KataGo ----------------
if [ "$SMOKE" = "1" ]; then
  log "SMOKE: synth data + nanogo-as-teacher"
  uv run python - <<'PY'
import os, numpy as np, torch
from dataclasses import asdict
os.makedirs("smoke_data", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
for k in range(4):
    full = np.zeros((8, 22, 19, 19), dtype=np.uint8)
    full[:, 0, :9, :9] = 1                       # 9x9 on-board
    full[:, 1, 1, 1] = 1; full[:, 1, 1, 3] = 1   # own (black) stones
    full[:, 2, 5, 5] = 1; full[:, 2, 5, 3] = 1   # opp (white) stones, disjoint
    packed = np.packbits(full.reshape(8, 22, 361), axis=2)
    glob = np.zeros((8, 19), dtype=np.float32); glob[:, 5] = 7.5 / 20.0
    np.savez(f"smoke_data/s{k}.npz", binaryInputNCHWPacked=packed, globalInputNC=glob)
from nanogo.net.model import Model, ModelConfig
from nanogo.go import features as F
cfg = ModelConfig.plain(8, 2); m = Model(cfg)
torch.save({"model": m.state_dict(), "optimizer": {}, "model_config": asdict(cfg),
            "step": 0, "spatial_subset": F.SPATIAL_SUBSET, "global_subset": F.GLOBAL_SUBSET},
           "checkpoints/teacher_tiny.pt")
print("synth ready")
PY
  log "SMOKE: relabel (teacher = nanogo)"
  uv run python scripts/relabel.py --src smoke_data --out smoke_distilled --n-files 4 --visits 1 \
    --teacher "uv run python scripts/run_engine.py -model checkpoints/teacher_tiny.pt -device cpu"
  log "SMOKE: train"
  uv run python scripts/train.py --data smoke_distilled --arch "$ARCH" --device cpu \
    --batch-size 16 --max-steps 30 --warmup 5 --eval-interval 20 --save-interval 30 \
    --val-files 1 --out checkpoints/smoke.pt
  log "SMOKE: judge (nanogo vs nanogo)"
  uv run python scripts/vs.py \
    --black "uv run python scripts/run_engine.py -model checkpoints/smoke.pt -device cpu" \
    --white "uv run python scripts/run_engine.py -model checkpoints/teacher_tiny.pt -device cpu" \
    --visits 4 --board 9 --komi 5.5 --max-moves 12
  log "SMOKE OK"
  exit 0
fi

# ---------------- REAL: GPU path ----------------
log "Install KataGo ($KATAGO_URL)"
mkdir -p katago_bin
curl -L "$KATAGO_URL" -o /tmp/katago.zip
unzip -o /tmp/katago.zip -d katago_bin >/dev/null
KATAGO=$(find katago_bin -name katago -type f | head -1)
chmod +x "$KATAGO"
cat > katago_bin/analysis.cfg <<'CFG'
# minimal analysis config; KataGo uses sensible defaults for everything else
numAnalysisThreads = 12
numSearchThreads = 12
nnMaxBatchSize = 256
CFG
"$KATAGO" version || { echo "KataGo failed to run — check KATAGO_URL matches the pod CUDA/TensorRT"; exit 1; }

log "Download models (b18 teacher + b6c96 opponent)"
mkdir -p models
curl -L "$TEACHER_URL" -o models/teacher.bin.gz
curl -L "$B6_URL" -o models/b6c96.bin.gz

log "Download training data ($DATA_DAYS days from $DATA_FROM)"
uv run python scripts/download_data.py --n "$DATA_DAYS" --from "$DATA_FROM"

KCFG="-config $PWD/katago_bin/analysis.cfg"
log "Relabel $RELABEL_FILES files with b18 (logit forcing)"
uv run python scripts/relabel.py --src data --out distilled --n-files "$RELABEL_FILES" --visits 1 \
  --teacher "$KATAGO analysis -model models/teacher.bin.gz $KCFG"

log "Train ($ARCH, $TRAIN_STEPS steps)"
uv run python scripts/train.py --data distilled --arch "$ARCH" --max-steps "$TRAIN_STEPS" \
  --eval-interval 2000 --save-interval 5000 --val-files 8 --out checkpoints/distill.pt

log "Judge vs b6c96 (b18 judge)"
uv run python scripts/vs.py \
  --black "uv run python scripts/run_engine.py -model checkpoints/distill.pt" \
  --white "$KATAGO analysis -model models/b6c96.bin.gz $KCFG" \
  --judge "$KATAGO analysis -model models/teacher.bin.gz $KCFG" \
  --visits 64 --board 19 --komi 7.5 --max-moves 200 --judge-visits 256
log "DONE"

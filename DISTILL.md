# Distilling more games on a remote pod

How to scale up the b18 → nanogo distillation on a cloud GPU (e.g. RunPod). The student is
tiny; the point of the GPU is **fast KataGo teacher inference** (relabeling millions of
positions) and fast training. See `experiments/SUMMARY.md` for why distillation is the path.

## 0. Provision

- **GPU**: RTX 4090 ($/perf sweet spot) is plenty; the student is ~1M params, the teacher
  (b18, 93 MB) is small. Bigger (L40S/A100) only speeds relabeling.
- **Disk**: distilled output is **~2 KB/position** → 1M ≈ 2 GB, 10M ≈ 20 GB, ~27M ≈ 55 GB.
  Plus source archives in flight (deletable after relabel) and models. **Provision ~100 GB.**
- A CUDA 12.x image with cuDNN/TensorRT available makes the KataGo install easy.

## 1. One command (bootstrap)

`scripts/runpod_bootstrap.sh` does the whole pipeline (clone → uv → KataGo → models → download
→ relabel → train → judge). Scale it with env vars:

```bash
export RELABEL_FILES=190000     # ~52 positions/file -> ~10M distilled positions
export DATA_DAYS=6              # source archives to download (each ~2.5M positions / ~1.4 GB)
export TRAIN_STEPS=150000
export ARCH=b6c96-gpool         # or b6c128nbt once the nbt block lands
curl -sSL https://raw.githubusercontent.com/sanderland/nanogo/dev/scripts/runpod_bootstrap.sh | bash
# KataGo install: run `bash scripts/setup_katago.sh` (CUDA+cuDNN build, handles the pod gotchas;
# the trt10.6.0-cuda12.6 release asset 404s). It emits katago_bin/katago.sh as the engine.
```

Smoke-test the orchestration first (CPU, no GPU/KataGo): `SMOKE=1 bash scripts/runpod_bootstrap.sh`.

## 2. Step by step (if you want control)

```bash
git clone -b dev https://github.com/sanderland/nanogo.git && cd nanogo
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev && uv run pytest -q          # katago-submodule tests skip; rest must pass

# KataGo + b18 teacher.  Use scripts/setup_katago.sh — it handles the gotchas on a clean CUDA
# pod (the trt10.6.0-cuda12.6 release asset 404s; pip TensorRT 10.2 is broken; no FUSE for the
# AppImage; cuDNN not preinstalled). It downloads the b18 teacher and emits a wrapper:
bash scripts/setup_katago.sh
KATAGO="$PWD/katago_bin/katago.sh"          # use this as the teacher/opponent/judge engine
printf 'numAnalysisThreads=12\nnnMaxBatchSize=256\n' > analysis.cfg
ls models/b18.bin.gz                          # downloaded by setup_katago.sh

# source games -> relabel with b18 (visits=1 = raw policy; strong enough to distill)
uv run python scripts/download_data.py --n 6 --from 2021-06-01
uv run python scripts/relabel.py --src data --out distilled --n-files 190000 --visits 1 \
  --teacher "$KATAGO analysis -model models/b18.bin.gz -config analysis.cfg"

# train the student
uv run python scripts/train.py --data distilled --arch b6c96-gpool \
  --max-steps 150000 --eval-interval 2000 --save-interval 10000 --val-files 16 \
  --out checkpoints/distill_big.pt
```

## 3. Scaling knobs

| want | change |
|------|--------|
| more distilled positions | `--n-files` on relabel (positions ≈ files × ~52); `--n` days to have enough source |
| stronger teacher targets | swap the b18 model for b28/b40c768 (bigger, slower); or **ensemble** (see below) |
| bigger / better student | `--arch` (e.g. `b6c128nbt`, `b10c128`); more `--max-steps` |
| keep disk down | delete `data/` after relabel (distilled npz already contain the input features) |

`--visits 1` is the right default for distillation (raw teacher policy). Don't spend visits on
the teacher; if you want stronger targets, **ensemble multiple nets** instead (average b18+b28+
b40 policies — a future `relabel.py --teacher` list) — more value per FLOP than deeper search.

## 4. Measure + retrieve

```bash
# judge vs KataGo b6c96 with b18 as neutral judge
curl -L https://katagoarchive.org/g170/neuralnets/g170-b6c96-s175395328-d26788732.bin.gz -o models/b6c96.bin.gz
uv run python scripts/vs.py \
  --black "uv run python scripts/run_engine.py -model checkpoints/distill_big.pt" \
  --white "$KATAGO analysis -model models/b6c96.bin.gz -config analysis.cfg" \
  --judge "$KATAGO analysis -model models/b18.bin.gz -config analysis.cfg" \
  --visits 64 --board 19 --max-moves 200 --judge-visits 256
# net-quality (no search, low noise):
uv run python scripts/policy_eval.py --src data --n 1000 \
  --ref "$KATAGO analysis -model models/b18.bin.gz -config analysis.cfg" \
  --engine "ours=uv run python scripts/run_engine.py -model checkpoints/distill_big.pt"
```

Copy the checkpoint back: `runpodctl send checkpoints/distill_big.pt` (or `scp`). It's ~13 MB.

## Notes

- Diagnostics show the in-game gap to b6c96 is currently **search**, not the net — distilling
  more games improves the net, but also run the search improvements (cpuct/LCB) to convert net
  quality into game strength.
- `relabel.py` pipelines ~52 queries per file; on a fast GPU you may want to raise KataGo's
  `numAnalysisThreads` / batch for higher relabel throughput.

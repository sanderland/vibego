#!/usr/bin/env bash
# Set up KataGo so it "just works" on a clean CUDA pod (no FUSE, no TensorRT/cuDNN preinstalled).
#
# What this handles, learned the hard way on a clean RunPod-style box:
#   - GPU BACKEND MUST MATCH THE GPU. On Blackwell (RTX PRO/50-series, compute capability sm_120)
#     the cuda12.5 build runs ~100x too slow (b18 benchmarked at ~25 visits/s — it crawls in
#     compatibility mode because CUDA 12.5 predates Blackwell). The cuda12.8 build runs natively:
#     ~31x faster (relabel ~26 -> ~800 files/min). So we use the cuda12.8-cudnn9.8.0 asset.
#     (The DISTILL.md trt10.6.0-cuda12.6 asset 404s; a trt10.9.0-cuda12.8 build also exists and
#      would be faster still, but needs TensorRT libs — the cuda12.8 build is the simpler win.)
#   - KataGo ships as an AppImage; containers usually have no FUSE -> must --appimage-extract
#     and run the extracted binary directly.
#   - cuDNN + unzip aren't present -> install cuDNN from a pip wheel, apt-get unzip.
#   - The binary needs libzip/libtcmalloc (bundled in the AppImage) + libcudnn + CUDA libs on
#     LD_LIBRARY_PATH -> emit a wrapper (katago_bin/katago.sh) that sets it all and execs katago.
#
# After this runs, use the teacher as:  REPO/katago_bin/katago.sh analysis -model ... -config ...
set -euo pipefail
log() { printf '\n=== %s ===\n' "$*"; }
REPO="${REPO:-/workspace/nanogo}"
cd "$REPO"

KG_VERSION="${KG_VERSION:-v1.16.4}"
KG_ASSET="${KG_ASSET:-katago-v1.16.4-cuda12.8-cudnn9.8.0-linux-x64.zip}"  # cuda12.8 = native Blackwell
CUDNN_PKG="${CUDNN_PKG:-nvidia-cudnn-cu12==9.8.0.87}"                     # matches the cudnn9.8 build
PYENV="${PYENV:-/opt/trt}"   # throwaway venv that just holds the cuDNN .so files
# Two FIXED teacher nets, pinned for reproducible distillation experiments:
#  - b18: the LAST kata1 b18c384nbt (s9996604416), final b18 before the run moved to b28. Small/fast.
#  - zhizi: kata1-zhizi-b40c768nbt-fdx6d (hzy / ZhiziGo), current strongest net. Big/slow, strongest targets.
#    "zhizi" keeps improving upstream; we pin this exact snapshot so our experiments stay comparable.
B18_URL="${B18_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}"
ZHIZI_URL="${ZHIZI_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-zhizi-b40c768nbt-fdx6d.bin.gz}"

command -v unzip >/dev/null 2>&1 || { log "apt-get unzip"; apt-get update -q && apt-get install -y -q unzip; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { log "install uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }

log "cuDNN wheel into $PYENV"
[ -d "$PYENV" ] || uv venv "$PYENV"
uv pip install --python "$PYENV/bin/python" "$CUDNN_PKG"

log "Download KataGo $KG_VERSION ($KG_ASSET)"
mkdir -p katago_cuda models
# GitHub's release CDN sometimes 504s on plain curl; gh routes around it. Fall back to curl.
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  rm -f "/tmp/$KG_ASSET"   # old gh (2.4) lacks --clobber; clear any stale file first
  gh release download "$KG_VERSION" --repo lightvector/KataGo --pattern "$KG_ASSET" --dir /tmp
else
  curl -fL "https://github.com/lightvector/KataGo/releases/download/$KG_VERSION/$KG_ASSET" -o "/tmp/$KG_ASSET"
fi
unzip -o "/tmp/$KG_ASSET" -d katago_cuda >/dev/null

log "Extract AppImage (no FUSE in containers)"
APPIMG="$(find katago_cuda -name katago -type f | head -1)"
chmod +x "$APPIMG"
( cd katago_cuda && ./"$(basename "$APPIMG")" --appimage-extract >/dev/null )

log "Write wrapper katago_bin/katago.sh + analysis.cfg"
mkdir -p katago_bin
cat > katago_bin/katago.sh <<'WRAP'
#!/usr/bin/env bash
# KataGo CUDA+cuDNN wrapper: extracted AppImage + pip cuDNN + CUDA libs on LD_LIBRARY_PATH.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KGROOT="$REPO/katago_cuda/squashfs-root"
CUDNN_LIB="$(dirname "$(find /opt/trt -name 'libcudnn.so.9' | head -1)")"  # cudnn9 = .so.9
export LD_LIBRARY_PATH="$KGROOT/usr/lib:$CUDNN_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec "$KGROOT/usr/bin/katago" "$@"
WRAP
chmod +x katago_bin/katago.sh
cat > katago_bin/analysis.cfg <<'CFG'
# minimal analysis config; KataGo uses sensible defaults for everything else
numAnalysisThreads = 12
numSearchThreads = 12
nnMaxBatchSize = 256
CFG
cat > katago_bin/relabel.cfg <<'CFG'
# high-throughput config for relabel.py (visits=1 raw policy): many independent 1-visit queries,
# so crank analysis threads and skip per-query search threads. ~800 b18 files/min on Blackwell.
numAnalysisThreads = 64
numSearchThreads = 1
nnMaxBatchSize = 128
CFG

dl() { [ -f "$2" ] || { log "Download $(basename "$2")"; curl -fL "$1" -o "$2"; }; }
dl "$B18_URL"   models/b18.bin.gz     # last fixed b18c384nbt  (small, fast)
dl "$ZHIZI_URL" models/zhizi.bin.gz   # zhizi-b40c768nbt-fdx6d (big, strongest)

log "Verify"
./katago_bin/katago.sh version
echo "OK -> teachers under models/:  b18.bin.gz | zhizi.bin.gz"
echo "     e.g. $REPO/katago_bin/katago.sh analysis -model models/b18.bin.gz -config katago_bin/analysis.cfg"

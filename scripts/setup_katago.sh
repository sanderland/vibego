#!/usr/bin/env bash
# Set up KataGo so it "just works" on a clean CUDA pod (no FUSE, no TensorRT/cuDNN preinstalled).
#
# What this handles, learned the hard way on a clean RunPod-style box:
#   - The DISTILL.md trt10.6.0-cuda12.6 asset DOESN'T EXIST for v1.16.4 (404). And the pip
#     TensorRT 10.2 wheel is broken (requests Windows-named libs/symbols: libnvinfer_builder_
#     resource_win.so / getNbCaskKLibs_INTERNAL_WIN). So we use the CUDA+cuDNN build instead.
#   - KataGo ships as an AppImage; containers usually have no FUSE -> must --appimage-extract
#     and run the extracted binary directly.
#   - cuDNN 8.9.7 + unzip aren't present -> install cuDNN from a pip wheel, apt-get unzip.
#   - The binary needs libzip/libtcmalloc (bundled in the AppImage) + libcudnn + CUDA libs on
#     LD_LIBRARY_PATH -> emit a wrapper (katago_bin/katago.sh) that sets it all and execs katago.
#
# After this runs, use the teacher as:  REPO/katago_bin/katago.sh analysis -model ... -config ...
set -euo pipefail
log() { printf '\n=== %s ===\n' "$*"; }
REPO="${REPO:-/workspace/nanogo}"
cd "$REPO"

KG_VERSION="${KG_VERSION:-v1.16.4}"
KG_ASSET="${KG_ASSET:-katago-v1.16.4-cuda12.5-cudnn8.9.7-linux-x64.zip}"  # works under driver CUDA>=12.5
CUDNN_PKG="${CUDNN_PKG:-nvidia-cudnn-cu12==8.9.7.29}"                     # matches the cudnn8.9.7 build
PYENV="${PYENV:-/opt/trt}"   # throwaway venv that just holds the cuDNN .so files
TEACHER_URL="${TEACHER_URL:-https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}"

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
  gh release download "$KG_VERSION" --repo lightvector/KataGo --pattern "$KG_ASSET" --dir /tmp --clobber
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
CUDNN_LIB="$(dirname "$(find /opt/trt -name 'libcudnn.so.8' | head -1)")"
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

[ -f models/teacher.bin.gz ] || { log "Download b18 teacher"; curl -fL "$TEACHER_URL" -o models/teacher.bin.gz; }

log "Verify"
./katago_bin/katago.sh version
echo "OK -> teacher: $REPO/katago_bin/katago.sh analysis -model models/teacher.bin.gz -config katago_bin/analysis.cfg"

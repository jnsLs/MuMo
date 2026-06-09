#!/usr/bin/env bash
#
# setup_mumo_env.sh — (re)build the `mumo` env and apply the GPU fixes this
# machine needs (RTX 5070 Ti / Blackwell sm_120, CUDA 13.0 driver).
#
# What it does:
#   1. Removes the existing (here: corrupted) `mumo` env.
#   2. Recreates it from environment.yml.
#   3. Upgrades the in-env CUDA toolkit to 13.0 so DeepSpeed can compile its ops.
#   4. Replaces the CPU/12.x torch with a Blackwell-capable cu130 build.
#   5. Makes `micromamba activate mumo` auto-export CUDA_HOME.
#   6. Verifies: GPU visible (sm_120), and DeepSpeedCPUAdam compiles.
#
# Why these steps: a plain `environment.yml` rebuild yields a CPU-only / CUDA-12.x
# torch that does NOT support the 5070 Ti (-> "GPU not recognized"), and an in-env
# cuda-toolkit that mismatches torch's CUDA (-> DeepSpeed CUDAMismatchException).
#
# Usage:
#   bash setup_mumo_env.sh                # rebuild from scratch (removes old env)
#   SKIP_REMOVE=1 bash setup_mumo_env.sh  # keep existing env, only (re)apply fixes
#
set -euo pipefail

ENV_NAME=mumo
TORCH_SPEC="torch==2.10.0+cu130"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
CUDA_TOOLKIT_VER="13.0"

# --- locate micromamba ---------------------------------------------------------
if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/micromamba/etc/profile.d/micromamba.sh"
elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
else
    echo "ERROR: micromamba not found (expected ~/micromamba)." >&2
    exit 1
fi
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f environment.yml ] || { echo "ERROR: environment.yml not found in $SCRIPT_DIR" >&2; exit 1; }

# --- 1. remove old env ---------------------------------------------------------
if [ "${SKIP_REMOVE:-0}" != "1" ]; then
    echo ">>> [1/6] Removing existing '$ENV_NAME' env (if present)..."
    micromamba env remove -y -n "$ENV_NAME" 2>/dev/null || true
fi

# --- 2. create from environment.yml -------------------------------------------
if ! micromamba env list | grep -qE "^\s*${ENV_NAME}\s|/${ENV_NAME}\$"; then
    echo ">>> [2/6] Creating '$ENV_NAME' from environment.yml (this takes a few minutes)..."
    micromamba env create -y -f environment.yml
else
    echo ">>> [2/6] Env '$ENV_NAME' already exists — skipping create (SKIP_REMOVE mode)."
fi

# --- 3. match CUDA toolkit to the driver / torch (13.0) ------------------------
echo ">>> [3/6] Installing cuda-toolkit=$CUDA_TOOLKIT_VER into '$ENV_NAME'..."
micromamba install -y -n "$ENV_NAME" -c conda-forge "cuda-toolkit=${CUDA_TOOLKIT_VER}"

# --- 4. Blackwell-capable torch (cu130) ---------------------------------------
echo ">>> [4/6] Installing $TORCH_SPEC (replaces the CPU/12.x torch)..."
micromamba run -n "$ENV_NAME" pip install --index-url "$TORCH_INDEX" "$TORCH_SPEC"

# --- 5. auto-export CUDA_HOME on activate -------------------------------------
echo ">>> [5/6] Adding CUDA_HOME activation hook..."
ACT_DIR="$MAMBA_ROOT/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
cat > "$ACT_DIR/zz_cuda_home.sh" <<'HOOK'
# Point DeepSpeed/nvcc at this env's CUDA toolkit
export CUDA_HOME="$CONDA_PREFIX"
HOOK

# --- 6. verify -----------------------------------------------------------------
echo ">>> [6/6] Verifying GPU + DeepSpeed op compilation..."
CUDA_HOME="$MAMBA_ROOT/envs/$ENV_NAME" micromamba run -n "$ENV_NAME" python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
print("arch list:", torch.cuda.get_arch_list())
if ok:
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    assert any("120" in a for a in torch.cuda.get_arch_list()), "WARNING: sm_120 (Blackwell) not in arch list"
    x = torch.randn(512, 512, device="cuda"); float((x @ x).sum())
    print("GPU matmul: OK")
else:
    raise SystemExit("ERROR: CUDA not available after install")
PY

echo ">>> Compiling DeepSpeedCPUAdam (first time only, ~1 min)..."
if CUDA_HOME="$MAMBA_ROOT/envs/$ENV_NAME" micromamba run -n "$ENV_NAME" python - <<'PY'
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
p = torch.nn.Parameter(torch.randn(8))
opt = DeepSpeedCPUAdam([p], lr=1e-3); p.grad = torch.randn(8); opt.step()
print("DeepSpeedCPUAdam: OK")
PY
then
    echo ""
    echo "============================================================"
    echo " SUCCESS — '$ENV_NAME' is rebuilt and GPU-ready."
    echo " Next:   micromamba activate $ENV_NAME   (auto-sets CUDA_HOME)"
    echo "         source init_env.sh"
    echo "         bash scripts/sft_moleculenet/regression/lipo.sh ..."
    echo " Note: this single-GPU box still can't fit the full 529M-param"
    echo "       fp32 finetune — run real lipo training on the HPC."
    echo "============================================================"
else
    echo "WARNING: DeepSpeedCPUAdam compile failed. Check CUDA_HOME / nvcc." >&2
    echo "         torch+GPU are fine; only the CPU-offload optimizer is affected." >&2
fi

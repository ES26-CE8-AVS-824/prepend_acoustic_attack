set -euo pipefail

#################################################################
PROJECT_ROOT="/home/peppermint/Aalborg/CE8/whispers-in-the-storm"
#################################################################

DATA_ROOT="$PROJECT_ROOT/data"
SINGULARITY_CACHE="$HOME/.singularity"

PREPEND_ATTACK_DIR="$PROJECT_ROOT/extern/prepend_acoustic_attack"

CU130_CONTAINER="$PROJECT_ROOT/sgmse_env_cu130_v7.sif"

singularity exec --nv --cleanenv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${SINGULARITY_CACHE}:/scratch/singularity" \
    "${CU130_CONTAINER}" \
    /bin/bash << CONTAINER_EOF
set -euo pipefail
# ── Writable cache directories ──────────────────────────────────────────────
export TMPDIR=/scratch/singularity/tmp
export NUMBA_CACHE_DIR=/scratch/singularity/tmp/numba_cache
export MPLCONFIGDIR=/scratch/singularity/tmp/matplotlib_cache
export TORCH_EXTENSIONS_DIR=/scratch/singularity/tmp/torch_extensions
export TORCH_KERNEL_CACHE=/scratch/singularity/tmp/torch_kernels
mkdir -p \$NUMBA_CACHE_DIR \$MPLCONFIGDIR \$TORCH_EXTENSIONS_DIR \$TORCH_KERNEL_CACHE

# ── General environment ──────────────────────────────────────────────────────
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE=${DATA_ROOT}/hf_cache

# ── Attacking ────────────────────────────────────────────────────────────────
printf "\n============ Generating prepend segment ============\n\n"
cd "${PREPEND_ATTACK_DIR}"
python process.py \
    --attack_model_path "${PREPEND_ATTACK_DIR}/experiments/vctk_ /whisper-tiny/transcribe/en/attack_train/audio-raw/attack_size5120/clip_val-1/prepend_attack_models/epoch11/model.th" \
    --save_path "${PREPEND_ATTACK_DIR}/audio_attack_segments/vctk_mute_tiny_epoch11.np.npy"
echo "Segment generated."
CONTAINER_EOF
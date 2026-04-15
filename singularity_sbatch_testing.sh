#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test/%j.out
#SBATCH --error=test/%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="/ceph/project/es26-ce8-avs-824/whispers-in-the-storm"
SINGULARITY_CACHE="$HOME/.singularity"
PREPEND_ATTACK_DIR="$PROJECT_ROOT/extern/prepend_acoustic_attack"
CU130_CONTAINER="$PROJECT_ROOT/sgmse_env_cu130_v3.sif"
DATA_ROOT="$PROJECT_ROOT/data"

mkdir -p "$PROJECT_ROOT/logs"

run_in_prepend_attack_venv() {
    local CMD="$1"
    (
    cd "${PREPEND_ATTACK_DIR}"
    singularity exec --nv \
        -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        -B "${SINGULARITY_CACHE}:/scratch/singularity" \
        "${CU130_CONTAINER}" \
        /bin/bash -c "
            set -euo pipefail && \
            export TMPDIR=/scratch/singularity/tmp && \
            export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
            export HF_DATASETS_CACHE=${DATA_ROOT}/hf_cache && \
            ${CMD}
        "
    )
}

run_in_prepend_attack_venv "pip list"
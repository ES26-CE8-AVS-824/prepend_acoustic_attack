#!/bin/bash

#SBATCH --job-name=t2t_fr
#SBATCH --output=logs/marko_t2t_train_%j.out
#SBATCH --error=logs/marko_t2t_train_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="/ceph/project/es26-ce8-avs-824/whispers-in-the-storm"
SINGULARITY_CACHE="$HOME/.singularity"
PREPEND_ATTACK_DIR="$PROJECT_ROOT/extern/prepend_acoustic_attack"
CU130_CONTAINER="$PROJECT_ROOT/sgmse_env_cu130_v7.sif"
DATA_ROOT="$PROJECT_ROOT/data"

mkdir -p "$PROJECT_ROOT/logs"

run_in_prepend_attack_venv() {
    local CMD="$1"
    (
    cd "${PREPEND_ATTACK_DIR}"
    singularity exec --nv --cleanenv --no-home \
        -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        -B "${SINGULARITY_CACHE}:/scratch/singularity" \
        --env PYTHONNOUSERSITE=1 \
        "${CU130_CONTAINER}" \
        /bin/bash -c "
            set -euo pipefail && \
            export TMPDIR=/scratch/singularity/tmp && \
            export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
            export HF_HOME=${DATA_ROOT}/hf_cache && \
            export XDG_CACHE_HOME=${DATA_ROOT}/hf_cache && \
            export MPLCONFIGDIR=/scratch/singularity/tmp && \
            ${CMD}
        "
    )
}

run_in_prepend_attack_venv "python train_attack.py \
    --model_name whisper-small-multi \
    --task transcribe \
    --language fr_en \
    --data_name vctk \
    --attack_method audio-raw \
    --attack_token transcribe \
    --attack_command translate \
    --max_epochs 40 \
    --save_freq 5 \
    --seed 42 \
    --attack_size 10240 \
    --clip_val 0.02 \
    --attack_init random \
    --bs 12 \
    --lr 1e-3
"
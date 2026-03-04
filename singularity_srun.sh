srun singularity exec --nv --writable-tmpfs \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/prepend_acoustic_attack/prepend_attack_venv:/scratch/prepend_attack_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/python/python_3.9.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/prepend_attack_venv/bin/activate && \
                   pip install -r requirements_marko.txt --no-cache-dir"
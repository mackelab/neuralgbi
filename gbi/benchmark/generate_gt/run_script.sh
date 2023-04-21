# #!/bin/bash
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=64
# #SBATCH --nodes=1
# #SBATCH --partition=cpu-long
# #SBATCH --mem-per-cpu=10G
# #SBATCH --output=./slurm_logs/%j_output.out
# #SBATCH --error=./slurm_logs/%j_error.err
# #SBATCH --time=72:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=richard.gao@uni-tuebingen.de

# scontrol show job $SLURM_JOB_ID
# python run_two_moons.py --multirun hydra/launcher=my_submitit_slurm task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified='specified','misspecified' task.is_known='known','unknown' task.beta=10,100,1000 task.name=two_moons

# # hydra/launcher=submitit_slurm hydra.launcher.timeout_min=300
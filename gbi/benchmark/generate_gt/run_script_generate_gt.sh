# ## generate xos
# python generate_xo.py 'uniform_1d' 10
# python generate_xo.py 'two_moons' 10
# python generate_xo.py 'linear_gaussian' 10
# python generate_xo.py 'gaussian_mixture' 10

# ## generate gt
# python run_uniform_1d.py --multirun hydra/launcher=my_submitit_slurm task.beta=4,20,100 task.name=uniform_1d
# python run_two_moons.py --multirun hydra/launcher=my_submitit_slurm task.beta=10,100,1000 task.name=two_moons
# python run_linear_gaussian.py --multirun hydra/launcher=my_submitit_slurm task.beta=1,10,100 task.name=linear_gaussian
# python run_gaussian_mixture.py --multirun hydra/launcher=my_submitit_slurm task.beta=2,10,50 task.name=gaussian_mixture

python run_two_moons.py --multirun hydra/launcher=my_submitit_slurm task.xo_index=1,6 task.is_specified='misspecified' task.is_known='known','unknown' task.beta=1000 task.name=two_moons

## THEN GO TO RUN_ALGORITHMS FOR TRAINING AND INFERENCE
## generate xos
# python ../tasks/generate_xo.py 'uniform_1d' 10
# python ../tasks/generate_xo.py 'two_moons' 10
# python ../tasks/generate_xo.py 'linear_gaussian' 10
# python ../tasks/generate_xo.py 'gaussian_mixture' 10


## generate gt
# python run_generate_gt.py --multirun hydra/launcher=my_submitit_slurm task.beta=4,20,100 task.name=uniform_1d
# python run_generate_gt.py --multirun hydra/launcher=my_submitit_slurm task.beta=10,100,1000 task.name=two_moons
# python run_generate_gt.py --multirun hydra/launcher=my_submitit_slurm task.beta=1,10,100 task.name=linear_gaussian
python run_generate_gt.py --multirun hydra/launcher=my_submitit_slurm task.beta=10,50,250 task.name=gaussian_mixture

# python run_generate_gt.py -m task.name=uniform_1d task.beta=4,20,100
# python run_generate_gt.py -m task.name=two_moons task.beta=10,100,1000
# python run_generate_gt.py -m task.name=linear_gaussian task.beta=1,10,100
# python run_generate_gt.py -m task.name=gaussian_mixture task.beta=10,50,250


## THEN GO TO RUN_ALGORITHMS FOR TRAINING AND INFERENCE


# betas:
# uniform_1d: 4, 20, 100
# two_moons: 10, 100, 1000
# linear_gaussian: 1, 10, 100
# gaussian_mixture: 10, 50, 250

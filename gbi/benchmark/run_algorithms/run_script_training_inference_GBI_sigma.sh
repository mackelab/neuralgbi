# ## TRAINING MODELS
# python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=GBI algorithm.noise_level=0. task.num_simulations=10000
python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=GBI algorithm.n_augmented_x=0 algorithm.train_with_obs=False task.num_simulations=1000

## do inference
TASK='uniform_1d'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=4,20,100


TASK='two_moons'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=10,100,1000


TASK='linear_gaussian'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1,10,100


TASK='gaussian_mixture'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=10,25,75
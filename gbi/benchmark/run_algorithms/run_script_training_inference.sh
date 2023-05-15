# ## TRAINING MODELS
python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=NPE,NLE,GBI,eGBI,ABC task.num_simulations=200

## do inference
TASK='uniform_1d'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=eGBI,GBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=4,20,100


TASK='two_moons'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=eGBI,GBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=10,100,1000


TASK='linear_gaussian'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=eGBI,GBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1,10,100


TASK='gaussian_mixture'
INF_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/ -t | head -n1`
# INF_DATETIME='2023_05_07__20_08_28' #'2023_05_03__18_41_31'
echo $INF_DATETIME
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=1
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=eGBI,GBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.beta=10,25,75
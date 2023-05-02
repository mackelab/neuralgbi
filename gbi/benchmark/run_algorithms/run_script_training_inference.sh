# ## TRAINING MODELS
# cd gbi/benchmark/run_algorithms/ 
# python run_training.py -m task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=NPE,NLE,GBI,eGBI,ABC
## OR 
# python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d,two_moons,gaussian_mixture algorithm=NPE,NLE,GBI,eGBI,ABC
# python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=linear_gaussian algorithm=NPE algorithm.sigmoid_theta=False
# python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=linear_gaussian algorithm=NLE,GBI,eGBI,ABC

python run_training.py -m task.name=two_moons task.num_simulations=10000 algorithm=GBI 

# ## do inference
INF_DATETIME='2023_04_21__22_38_19'
# TASK='uniform_1d'
# BETA1=4
# BETA2=20
# BETA3=100
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


# TASK='two_moons'
# BETA1=10
# BETA2=100
# BETA3=1000
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


TASK='linear_gaussian'
BETA1=1
BETA2=10
BETA3=100
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3
python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=eGBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


# TASK='gaussian_mixture'
# BETA1=10
# BETA2=50
# BETA3=250
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

python run_inference.py --multirun hydra/launcher=my_submitit_slurm algorithm=NPE,NLE trained_inference_datetime=2023_05_01__23_32_09 task.name=gaussian_mixture task.beta=1
# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


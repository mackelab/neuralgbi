# ## TRAINING MODELS
# cd gbi/benchmark/run_algorithms/ 
# python run_training.py -m task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=NPE,NLE,GBI,eGBI,ABC
## OR 
# python run_training.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d,two_moons,linear_gaussian,gaussian_mixture algorithm=NPE,NLE,GBI,eGBI,ABC


# ## do inference
# INF_DATETIME='2023_04_07__19_18_18'
# TASK='uniform_1d'
# BETA1=4
# BETA2=20
# BETA3=100
# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

# INF_DATETIME='2023_04_08__12_42_12'
# TASK='two_moons'
# BETA1=10
# BETA2=100
# BETA3=1000
# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

# INF_DATETIME='2023_04_08__12_42_12'
# TASK='linear_gaussian'
# BETA1=1
# BETA2=10
# BETA3=100
# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

# INF_DATETIME='2023_04_13__12_02_17'
# TASK='gaussian_mixture'
# BETA1=2
# BETA2=10
# BETA3=50
# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,eGBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

## collect posterior and metrics
#GT_DATETIME='YYYY_MM_DD__hh_mm_ss'
#INF_DATETIME='YYYY_MM_DD__hh_mm_ss'
#TASK=two_moons
# python collect_posterior_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME
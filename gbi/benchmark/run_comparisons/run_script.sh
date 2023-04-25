## collect posterior and metrics
# GT_DATETIME='2023_04_21__18_27_39'
# INF_DATETIME='2023_04_21__22_38_19'
# TASK=uniform_1d
# python collect_posterior_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME



# python collect_posterior_samples.py task.name=uniform_1d gt_datetime='None' inference_datetime='None'
# python collect_posterior_samples.py task.name=two_moons gt_datetime='None' inference_datetime='None'
# python collect_posterior_samples.py task.name=linear_gaussian gt_datetime='None' inference_datetime='None'
# python collect_posterior_samples.py task.name=gaussian_mixture gt_datetime='None' inference_datetime='None'


# python collect_posterior_samples.py task.name=two_moons gt_datetime='2023_04_20__18_40_28' inference_datetime='2023_04_21__22_38_19'

# python collect_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=uniform_1d inference_datetime='2023_04_21__22_38_19'





# save result of ls as a shell variable
# https://stackoverflow.com/questions/4651437/how-do-i-set-a-variable-to-the-output-of-a-command-in-bash
 


# ls -t | head -n1

GT_DATETIME=None
INF_DATETIME='2023_04_21__22_38_19'

TASK=gaussian_mixture
python run_collect_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME
COLLECTION_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/$INF_DATETIME/posterior_samples_collected -t | head -n1`
python run_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=$TASK collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME algorithm=GT,GBI,eGBI,NPE,NLE,ABC task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown
python run_metrics.py task.name=$TASK algorithm=COLLECT collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME 
GT_DATETIME=None
# '2023_05_11__23_45_10': 200
# '2023_05_03__18_41_31': 1000
# '2023_05_07__20_08_28': 10000
INF_DATETIME='2023_05_11__23_45_10'
for TASK in gaussian_mixture # uniform_1d two_moons linear_gaussian gaussian_mixture
do
    echo $TASK
    python run_collect_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME
    COLLECTION_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/$INF_DATETIME/posterior_samples_collected -t | head -n1`
    echo $COLLECTION_DATETIME
    python run_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=$TASK collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME algorithm=GT,GBI,eGBI,NPE,NLE,ABC
    python run_metrics.py task.name=$TASK algorithm=COLLECT collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME   
done

# save result of ls as a shell variable
# https://stackoverflow.com/questions/4651437/how-do-i-set-a-variable-to-the-output-of-a-command-in-bash 
# ls -t | head -n1

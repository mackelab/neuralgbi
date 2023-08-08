GT_DATETIME=None
# '2023_05_03__18_41_31': 1000
# '2023_05_07__20_08_28': 10000
# '2023_05_11__23_45_10': 200
# '2023_08_03__16_02_52': 10000 NLE_tempered
# '2023_08_04__15_14_42': 10000 GBI 5-sigma
# '2023_08_04__17_17_42': 10000 GBI 0-sigma
# '2023_08_05__16_45_05': 10000 GBI no extra
# '2023_08_05__23_24_44': 1000 GBI no extra

INF_DATETIME='2023_08_05__23_24_44'
for TASK in uniform_1d two_moons linear_gaussian gaussian_mixture
do
    echo $TASK
    # python run_collect_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME
    python run_collect_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME algos=['GBI']
    COLLECTION_DATETIME=`ls ../../../results/benchmark/algorithms/$TASK/$INF_DATETIME/posterior_samples_collected -t | head -n1`
    echo $COLLECTION_DATETIME
    # python run_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=$TASK collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME algorithm=GT,GBI,eGBI,NPE,NLE,ABC
    # python run_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=$TASK collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME algorithm=NLE_tempered compute_GBI_dist_prediction=False
    python run_metrics.py --multirun hydra/launcher=my_submitit_slurm task.name=$TASK collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME algorithm=GBI
    python run_metrics.py task.name=$TASK algorithm=COLLECT collection_datetime=$COLLECTION_DATETIME inference_datetime=$INF_DATETIME
done

# save result of ls as a shell variable
# https://stackoverflow.com/questions/4651437/how-do-i-set-a-variable-to-the-output-of-a-command-in-bash 
# ls -t | head -n1

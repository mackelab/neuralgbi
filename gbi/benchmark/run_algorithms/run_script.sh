
# # training
# python run_training.py -m task.name=two_moons,linear_gaussian,gaussian_mixture algorithm=NPE,NLE,ABC,eGBI,GBI
# # training fails for gaussian_mixture with all algorithms except ABC


# run inference
INF_DATETIME='2023_04_07__19_18_18'
TASK='uniform_1d'
BETA1=4
BETA2=20
BETA3=100

# INF_DATETIME='2023_04_08__12_42_12'
# TASK='two_moons'
# BETA1=10
# BETA2=100
# BETA3=1000

# INF_DATETIME='2023_04_08__12_42_12'
INF_DATETIME='2023_04_19__12_40_45'
TASK='linear_gaussian'
BETA1=0.1
BETA2=1.0
BETA3=10.0

# INF_DATETIME='2023_04_13__12_02_17'
# TASK='gaussian_mixture'
# BETA1=2.0
# BETA2=10.0
# BETA3=50.0


# python run_inference.py -m algorithm=NPE,NLE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=1
# python run_inference.py -m algorithm=GBI,ABC trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3

# python run_inference.py -m algorithm=NPE trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3
python run_inference.py -m algorithm=GBI trained_inference_datetime=$INF_DATETIME task.name=$TASK task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=$BETA1,$BETA2,$BETA3


# python run_inference.py algorithm=NPE trained_inference_datetime=2023_04_13__12_02_17 task.name=gaussian_mixture task.xo_index=0 task.is_specified=specified task.is_known=known task.beta=1

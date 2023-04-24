## collect posterior and metrics
# GT_DATETIME='2023_04_21__18_27_39'
# INF_DATETIME='2023_04_21__22_38_19'
# TASK=uniform_1d
# python collect_posterior_samples.py task.name=$TASK gt_datetime=$GT_DATETIME inference_datetime=$INF_DATETIME



python collect_posterior_samples.py task.name=uniform_1d gt_datetime='None' inference_datetime='None'
python collect_posterior_samples.py task.name=two_moons gt_datetime='None' inference_datetime='None'
python collect_posterior_samples.py task.name=linear_gaussian gt_datetime='None' inference_datetime='None'
python collect_posterior_samples.py task.name=gaussian_mixture gt_datetime='None' inference_datetime='None'
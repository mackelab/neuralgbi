
# run inference
INF_DATETIME='2023_04_07__17_57_00'
python run_inference.py algorithm=eGBI trained_inference_datetime=$INF_DATETIME task.name=uniform_1d task.xo_index=0 task.is_specified=specified task.is_known=unknown task.beta=4
# Hodgkin-Huxley model

## Compilation

In order to use the Cython version of the solver, a compilation step is needed.

For that, run the following (in this subfolder):
```
python compile.py build_ext --inplace
```


## Reproducing results

The following things have to be run to reproduce the results:

- generate the simulations: ```python simulate.py```
- train the GBI network: ```python train_gbi.py nsims=100000 num_layers=7 num_hidden=100 training_batch_size=5000```
- train the NPE network: ```python train_npe.py -m nsims=100000,1000000 training_batch_size=5000```
- generate the synthetic observations: ```python gen_synthetic_xo.py```
- GBI predictive samples: ```python gen_gbi_predictives.py -m observation="allen","synthetic"```
- NPE predictive samples: ```python gen_npe_predictives.py -m nsims=100000,1000000 observation="allen","synthetic"```

After that, you should be able to run the notebooks the lie in the `paper` folder.
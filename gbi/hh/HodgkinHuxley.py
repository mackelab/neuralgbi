import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator

def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x

def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x

class HodgkinHuxley(BaseSimulator):
    def __init__(self, I, dt, V0, cython=False, prior_log=False, reduced_model=False, seed=None):
        """Hodgkin-Huxley simulator
        Parameters
        ----------
        I : array
            Numpy array with the input I
        dt : float
            Timestep
        V0 : float
            Voltage at first time step
        cython : bool
            If True, will use cython version of simulator (different import)
        reduced_model : bool
            If True, model with 2 parameters instead of 8
        seed : int or None
            If set, randomness across runs is disabled
        """
        if reduced_model:
            dim_param = 2
        else:
            dim_param = 8

        super().__init__(dim_param=dim_param, seed=seed)
        self.I = I
        self.cython = cython
        self.dt = dt
        self.t = np.arange(0, len(self.I), 1)*self.dt
        self.prior_log = prior_log

        if cython:
            import model.HodgkinHuxleyBioPhysCython as bm
        else:
            import model.HodgkinHuxleyBioPhys as bm
        self.bm = bm

        # parameters that globally govern the simulations
        self.init = [V0]  # =V0

    def gen_single(self, params):
        """Forward model for simulator for single parameter set
        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector
        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = param_invtransform(self.prior_log,np.asarray(params))

        assert params.ndim == 1, 'params.ndim must be 1'

        hh_seed = self.gen_newseed()

        hh = self.bm.HH(self.init, params.reshape(1, -1), seed=hh_seed)
        states = hh.sim_time(self.dt, self.t, self.I)

        return {'data': states.reshape(-1),
                'time': self.t,
                'dt': self.dt,
                'I': self.I.reshape(-1)}

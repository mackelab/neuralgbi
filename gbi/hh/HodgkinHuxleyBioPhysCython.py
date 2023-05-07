import numpy as np

from . import biophys_cython_comp

solver = biophys_cython_comp.forwardeuler
# solver = biophys_cython_comp.backwardeuler
# solver = biophys_cython_comp.hinesmethod

# def solver(t, I, V, m, n, h, p, q, r, u, dt, r_mat)
# t: array of time steps
# I: array of I values
# V: array of V values (OUTPUT)
# m, n, h, p, q, r, u: buffers for gating variables
# dt: time step
# r_mat: array of random inputs (voltage noise)
#
# The arrays must have the same size. The simulation runs until V is exhausted.


class HH:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        self.seed = seed
        if seed is not None:
            biophys_cython_comp.seed(seed)
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I, fineness=1, max_n_steps=float("inf")):
        """Simulates the model for a specified time duration."""

        biophys_cython_comp.setparams(self.params)
        tstep = float(dt)

        # explictly cast everything to double precision
        t = t.astype(np.float64)
        I = I.astype(np.float64)
        V = np.zeros_like(t).astype(np.float64)  # baseline voltage
        V[0] = float(self.state[0])
        n = np.zeros_like(t).astype(np.float64)
        m = np.zeros_like(t).astype(np.float64)
        h = np.zeros_like(t).astype(np.float64)
        p = np.zeros_like(t).astype(np.float64)
        q = np.zeros_like(t).astype(np.float64)
        r = np.zeros_like(t).astype(np.float64)
        u = np.zeros_like(t).astype(np.float64)
        r_mat = self.rng.randn(len(t)).astype(np.float64)

        solver(t, I, V, m, n, h, p, q, r, u, tstep, r_mat)

        return np.array(V).reshape(-1, 1)

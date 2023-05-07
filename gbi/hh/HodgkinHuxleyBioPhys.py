import numpy as np


class HH:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I):
        """Simulates the model for a specified time duration."""

        if len(self.params[0, :]) == 2:
            gbar_Na = self.params[0, 0]  # mS/cm2
            gbar_Na.astype(float)
            gbar_K = self.params[0, 1]
            gbar_K.astype(float)
            g_leak = 0.1
            gbar_M = 0.07
            tau_max = 6e2  # ms
            Vt = -60.0
            nois_fact = 0.1
            E_leak = -70.0
        else:
            gbar_Na = self.params[0, 0]  # mS/cm2
            gbar_Na.astype(float)
            gbar_K = self.params[0, 1]
            gbar_K.astype(float)
            g_leak = self.params[0, 2]
            g_leak.astype(float)
            gbar_M = self.params[0, 3]
            gbar_M.astype(float)
            # gbar_L = self.params[0,4]
            # gbar_L.astype(float)
            # gbar_T = self.params[0,5]
            # gbar_T.astype(float)
            tau_max = self.params[0, 4]  # ms
            tau_max.astype(float)
            Vt = -self.params[0, 5]
            Vt.astype(float)
            nois_fact = self.params[0, 6]
            nois_fact.astype(float)
            E_leak = -self.params[0, 7]
            E_leak.astype(float)

        tstep = float(dt)

        # Parameters
        nois_fact_obs = 0.0
        C = 1.0  # uF/cm2
        E_Na = 53  # mV
        E_K = -107

        # to generate burst
        E_Ca = 120
        Vx = 2
        gbar_L = 0
        gbar_T = 0

        ####################################
        # kinetics
        def Exp(z):
            if z < -5e2:
                return np.exp(-5e2)
            else:
                return np.exp(z)

        def efun(z):
            if np.abs(z) < 1e-4:
                return 1 - z / 2
            else:
                return z / (Exp(z) - 1)

        def alpha_m(x):
            v1 = x - Vt - 13.0
            return 0.32 * efun(-0.25 * v1) / 0.25

        def beta_m(x):
            v1 = x - Vt - 40
            return 0.28 * efun(0.2 * v1) / 0.2

        def alpha_h(x):
            v1 = x - Vt - 17.0
            return 0.128 * Exp(-v1 / 18.0)

        def beta_h(x):
            v1 = x - Vt - 40.0
            return 4.0 / (1 + Exp(-0.2 * v1))

        def alpha_n(x):
            v1 = x - Vt - 15.0
            return 0.032 * efun(-0.2 * v1) / 0.2

        def beta_n(x):
            v1 = x - Vt - 10.0
            return 0.5 * Exp(-v1 / 40)

        # slow non-inactivating K+
        def p_inf(x):
            v1 = x + 35.0
            return 1.0 / (1.0 + Exp(-0.1 * v1))

        def tau_p(x):
            v1 = x + 35.0
            return tau_max / (3.3 * Exp(0.05 * v1) + Exp(-0.05 * v1))

        # to generate burst
        # high-threshold Ca2+
        def alpha_q(x):
            v1 = x + 27
            return 0.055 * efun(-v1 / 3.8) * 3.8

        def beta_q(x):
            v1 = x + 75
            return 0.94 * Exp(-v1 / 17)

        def alpha_r(x):
            v1 = x + 13
            return 0.000457 * Exp(-v1 / 50)

        def beta_r(x):
            v1 = x + 15
            return 0.0065 / (1 + Exp(-v1 / 28))

        # low-threshold Ca2+
        def s_inf(x):
            v1 = x + Vx + 57
            return 1 / (1 + Exp(-v1 / 6.2))

        def u_inf(x):
            v1 = x + Vx + 81
            return 1 / (1 + Exp(v1 / 4))

        def tau_u(x):
            v1 = x + Vx + 84
            v2 = x + Vx + 113.2
            return 30.8 / 3.7 + (211.4 + Exp(v2 / 5)) / (3.7 * (1 + Exp(v1 / 3.2)))

        def tau_n(x):
            return 1 / (alpha_n(x) + beta_n(x))

        def n_inf(x):
            return alpha_n(x) / (alpha_n(x) + beta_n(x))

        def tau_m(x):
            return 1 / (alpha_m(x) + beta_m(x))

        def m_inf(x):
            return alpha_m(x) / (alpha_m(x) + beta_m(x))

        def tau_h(x):
            return 1 / (alpha_h(x) + beta_h(x))

        def h_inf(x):
            return alpha_h(x) / (alpha_h(x) + beta_h(x))

        def tau_q(x):
            return 1 / (alpha_q(x) + beta_q(x))

        def q_inf(x):
            return alpha_q(x) / (alpha_q(x) + beta_q(x))

        def tau_r(x):
            return 1 / (alpha_r(x) + beta_r(x))

        def r_inf(x):
            return alpha_r(x) / (alpha_r(x) + beta_r(x))

        ####################################

        # simulation from initial point
        V = np.zeros_like(t)  # baseline voltage
        n = np.zeros_like(t)
        m = np.zeros_like(t)
        h = np.zeros_like(t)
        p = np.zeros_like(t)
        q = np.zeros_like(t)
        r = np.zeros_like(t)
        u = np.zeros_like(t)

        V[0] = float(self.state[0])
        # V[0] = E_leak
        n[0] = n_inf(V[0])
        m[0] = m_inf(V[0])
        h[0] = h_inf(V[0])
        p[0] = p_inf(V[0])
        q[0] = q_inf(V[0])
        r[0] = r_inf(V[0])
        u[0] = u_inf(V[0])

        for i in range(1, t.shape[0]):
            tau_V_inv = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1]
                + (n[i - 1] ** 4) * gbar_K
                + g_leak
                + gbar_M * p[i - 1]
                + gbar_L * (q[i - 1] ** 2) * r[i - 1]
                + gbar_T * (s_inf(V[i - 1]) ** 2) * u[i - 1]
            ) / C
            V_inf = (
                (m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na
                + (n[i - 1] ** 4) * gbar_K * E_K
                + g_leak * E_leak
                + gbar_M * p[i - 1] * E_K
                + gbar_L * (q[i - 1] ** 2) * r[i - 1] * E_Ca
                + gbar_T * (s_inf(V[i - 1]) ** 2) * u[i - 1] * E_Ca
                + I[i - 1]
                + nois_fact * self.rng.randn() / (tstep**0.5)
            ) / (tau_V_inv * C)
            V[i] = V_inf + (V[i - 1] - V_inf) * Exp(-tstep * tau_V_inv)
            n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * Exp(-tstep / tau_n(V[i]))
            m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * Exp(-tstep / tau_m(V[i]))
            h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * Exp(-tstep / tau_h(V[i]))
            p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * Exp(-tstep / tau_p(V[i]))
            q[i] = q_inf(V[i]) + (q[i - 1] - q_inf(V[i])) * Exp(-tstep / tau_q(V[i]))
            r[i] = r_inf(V[i]) + (r[i - 1] - r_inf(V[i])) * Exp(-tstep / tau_r(V[i]))
            u[i] = u_inf(V[i]) + (u[i - 1] - u_inf(V[i])) * Exp(-tstep / tau_u(V[i]))

        #        return np.array(V).reshape(-1,1)
        return np.array(V).reshape(-1, 1) + nois_fact_obs * self.rng.randn(
            t.shape[0], 1
        )

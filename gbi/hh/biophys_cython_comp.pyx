import numpy as np
cimport numpy as np
import scipy
from libc.math cimport exp, log, cos, sqrt
cimport cython

###############################################################################
# parameters
cdef double gbar_K = 13
cdef double gbar_Na = 10
cdef double g_leak = 0.1
cdef double E_K = -107
cdef double E_Na = 53
cdef double E_leak = -70
cdef double C = 1
cdef double Vt = -50.0
cdef double gbar_M = 0.1
cdef double tau_max = 1000

cdef double nois_fact = 0.5

cdef double gbar_L = 0.
cdef double E_Ca = 120
cdef double gbar_T = 0.
cdef double Vx = 2

###############################################################################
# kinetics
@cython.cdivision(True)
cdef double Exp(double z):
	if z < -5e2:
		return exp(-5e2)
	else:
		return exp(z)

@cython.cdivision(True)
cdef double efun(double z):
	if abs(z) < 1e-4:
		return 1 - z/2
	else:
		return z / (Exp(z) - 1)

@cython.cdivision(True)
cdef double alpha_n(double x):
	cdef double v1 = x - Vt - 15
	return 0.032 * efun(-0.2 * v1) / 0.2

@cython.cdivision(True)
cdef double beta_n(double x):
	cdef double v1 = x - Vt - 10
	return 0.5*Exp(-v1/40)

@cython.cdivision(True)
cdef double alpha_m(double x):
	cdef double v1 = x - Vt - 13
	return 0.32 * efun(-0.25*v1) / 0.25

@cython.cdivision(True)
cdef double beta_m(double x):
	cdef double v1 = x - Vt - 40
	return 0.28*efun(0.2*v1)/0.2

@cython.cdivision(True)
cdef double alpha_h(double x):
	cdef double v1 = x - Vt - 17
	return 0.128*Exp(-v1/18)

@cython.cdivision(True)
cdef double beta_h(double x):
	cdef double v1 = x - Vt - 40
	return 4.0/(1 + Exp(-0.2*v1))

@cython.cdivision(True)
cdef double p_inf(double x):
	cdef double v1 = x + 35
	return 1.0/(1 + Exp(-0.1*v1))

@cython.cdivision(True)
cdef double tau_p(double x):
	cdef double v1 = x + 35
	return tau_max/(3.3*Exp(0.05*v1) + Exp(-0.05*v1))

# to generate burst
# high-threshold Ca2+
@cython.cdivision(True)
cdef double alpha_q(double x):
	cdef double v1 = x + 27
	return 0.055*efun(-v1/3.8)*3.8

@cython.cdivision(True)
cdef double beta_q(double x):
	cdef double v1 = x + 75
	return 0.94*Exp(-v1/17)

@cython.cdivision(True)
cdef double alpha_r(double x):
	cdef double v1 = x + 13
	return 0.000457*Exp(-v1/50)

@cython.cdivision(True)
cdef double beta_r(double x):
	cdef double v1 = x + 15
	return 0.0065/(1 + Exp(-v1/28))

# low-threshold Ca2+
@cython.cdivision(True)
cdef double s_inf(double x):
	cdef double v1 = x + Vx + 57
	return 1/(1 + Exp(-v1/6.2))

@cython.cdivision(True)
cdef double u_inf(double x):
	cdef double v1 = x + Vx + 81
	return 1/(1 + Exp(v1/4))

@cython.cdivision(True)
cdef double tau_u(double x):
	cdef double v1 = x + Vx + 84
	cdef double v2 = x + Vx + 113.2
	return 30.8/3.7 + (211.4 + Exp(v2/5))/(3.7*(1 + Exp(v1/3.2)))

@cython.cdivision(True)
cdef double tau_gate(double a,double b):
	return 1/(a + b)

@cython.cdivision(True)
cdef double gate_inf(double a,double b):
	return a/(a + b)

def seed(n):
	np.random.seed(n)

def setnoisefactor(double x):
	global nois_fact
	nois_fact = x

def setparams(params):
	global gbar_Na, gbar_K, g_leak, gbar_M, gbar_L, gbar_T, tau_max, Vt, nois_fact, E_leak
	
	if len(params[0,:]) == 2:
		gbar_Na = params[0,0]
		gbar_K = params[0,1]
		g_leak = 0.1
		gbar_M = 0.07
		tau_max = 6e2
		Vt = -60.
		nois_fact = 0.1
		E_leak = -70.
		gbar_L = 0.
		gbar_T = 0.
	else:
		gbar_Na = params[0,0]
		gbar_K = params[0,1]
		g_leak = params[0,2]
		gbar_M = params[0,3]
		#gbar_L = params[0,4]
		#gbar_T = params[0,5]
		tau_max = params[0,4]
		Vt = -params[0,5]
		nois_fact = params[0,6]
		E_leak = -params[0,7]
		gbar_L = 0.
		gbar_T = 0.


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatebe(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u,int i,double tstep,int fineness):
	cdef np.ndarray[double,ndim=1] b = np.array([V[i-1],n[i-1],m[i-1],h[i-1],p[i-1],q[i-1],r[i-1],u[i-1]])
	cdef double dt = tstep / fineness

	for j in range(fineness):
		b = scipy.optimize.fixed_point(updatefunc,b,args=(b,I[i-1],dt))

	V[i] = b[0] + nois_fact * (np.random.normal() * sqrt(dt))
	n[i] = b[1]
	m[i] = b[2]
	h[i] = b[3]
	p[i] = b[4]
	q[i] = b[5]
	r[i] = b[6]
	u[i] = b[7]

# Backward Euler
def backwardeuler(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u,double tstep,int fineness):
	#V[0] = E_leak
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])
	q[0] = alpha_q(V[0])/(alpha_q(V[0])+beta_q(V[0]))
	r[0] = alpha_r(V[0])/(alpha_r(V[0])+beta_r(V[0]))
	u[0] = u_inf(V[0])

	for i in range(1, t.shape[0]):
		updatebe(I,V,m,n,h,p,q,r,u,i,tstep,fineness)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def updatefunc(np.ndarray[double,ndim=1] a,np.ndarray[double,ndim=1] b,double I,double tstep):
	cdef np.ndarray[double,ndim=1] ret = np.empty(5)

	# currents
	cdef double I_Na = (a[2]**3)*gbar_Na*a[3]*(a[0]-E_Na)
	cdef double I_K = (a[1]**4)*gbar_K*(a[0]-E_K)
	cdef double I_leak = g_leak*(a[0]-E_leak)
	cdef double I_M = gbar_M*a[4]*(a[0]-E_K)
	cdef double I_L = gbar_L*(a[5]**2)*a[6]*(a[0]-E_Ca)
	cdef double I_T = gbar_T*(s_inf(a[0])**2)*a[7]*(a[0]-E_Ca)
	cdef double I_ion = I - I_K - I_Na - I_leak - I_M - I_L - I_T

	###############################
	ret[0] = b[0] + tstep*(I_ion)/C
	ret[1] = b[1] + tstep*(alpha_n(a[0])*(1-a[1]) - beta_n(a[0])*a[1])
	ret[2] = b[2] + tstep*(alpha_m(a[0])*(1-a[2]) - beta_m(a[0])*a[2])
	ret[3] = b[3] + tstep*(alpha_h(a[0])*(1-a[3]) - beta_h(a[0])*a[3])
	ret[4] = b[4] + tstep*(p_inf(a[0])-a[4])/tau_p(a[0])
	ret[5] = b[5] + tstep*(alpha_q(a[0])*(1-a[5]) - beta_q(a[0])*a[5])
	ret[6] = b[6] + tstep*(alpha_r(a[0])*(1-a[6]) - beta_r(a[0])*a[6])
	ret[7] = b[7] + tstep*(u_inf(a[0])-a[7])/tau_u(a[0])

	return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatefe(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u,int i,double tstep,np.ndarray[double,ndim=1] r_mat):
	# currents
	cdef double cV = V[i-1]
	cdef double cm = m[i-1]
	cdef double cn = n[i-1]
	cdef double ch = h[i-1]
	cdef double cp = p[i-1]
	cdef double cq = q[i-1]
	cdef double cr = r[i-1]
	cdef double cu = u[i-1]

	cdef double tau_V_inv, V_inf, an, bn, am, bm, ah, bh, aq, bq, ar, br

	tau_V_inv = ( (cm**3)*gbar_Na*ch+(cn**4)*gbar_K+g_leak+gbar_M*cp
	            +gbar_L*(cq**2)*cr + gbar_T*(s_inf(cV)**2)*cu )/C
	V_inf = ( (cm**3)*gbar_Na*ch*E_Na+(cn**4)*gbar_K*E_K+g_leak*E_leak+gbar_M*cp*E_K
	        +gbar_L*(cq**2)*cr*E_Ca + gbar_T*(s_inf(cV)**2)*cu*E_Ca
					+I[i-1]+nois_fact*r_mat[i-1]/sqrt(tstep) )/(tau_V_inv*C)

	cV = V_inf + (cV-V_inf)*Exp(-tstep*tau_V_inv)

	an = alpha_n(cV)
	am = alpha_m(cV)
	ah = alpha_h(cV)
	aq = alpha_q(cV)
	ar = alpha_r(cV)

	bn = beta_n(cV)
	bm = beta_m(cV)
	bh = beta_h(cV)
	bq = beta_q(cV)
	br = beta_r(cV)

	cn = gate_inf(an,bn)+(cn-gate_inf(an,bn))*Exp(-tstep/tau_gate(an,bn))
	cm = gate_inf(am,bm)+(cm-gate_inf(am,bm))*Exp(-tstep/tau_gate(am,bm))
	ch = gate_inf(ah,bh)+(ch-gate_inf(ah,bh))*Exp(-tstep/tau_gate(ah,bh))
	cp = p_inf(cV)+(cp-p_inf(cV))*Exp(-tstep/tau_p(cV))
	cq = gate_inf(aq,bq)+(cq-gate_inf(aq,bq))*Exp(-tstep/tau_gate(aq,bq))
	cr = gate_inf(ar,br)+(cr-gate_inf(ar,br))*Exp(-tstep/tau_gate(ar,br))
	cu = u_inf(cV)+(cu-u_inf(cV))*Exp(-tstep/tau_u(cV))

	V[i] = cV
	n[i] = cn
	m[i] = cm
	h[i] = ch
	p[i] = cp
	q[i] = cq
	r[i] = cr
	u[i] = cu

# Forward Euler
def forwardeuler(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u, double tstep,np.ndarray[double,ndim=1] r_mat):
	cdef double an1, bn1, am1, bm1, ah1, bh1, aq1, bq1, ar1, br1

	#V[0] = E_leak

	an1 = alpha_n(V[0])
	am1 = alpha_m(V[0])
	ah1 = alpha_h(V[0])
	aq1 = alpha_q(V[0])
	ar1 = alpha_r(V[0])

	bn1 = beta_n(V[0])
	bm1 = beta_m(V[0])
	bh1 = beta_h(V[0])
	bq1 = beta_q(V[0])
	br1 = beta_r(V[0])

	n[0] = gate_inf(an1,bn1)
	m[0] = gate_inf(am1,bm1)
	h[0] = gate_inf(ah1,bh1)
	p[0] = p_inf(V[0])
	q[0] = gate_inf(aq1,bq1)
	r[0] = gate_inf(ar1,br1)
	u[0] = u_inf(V[0])

	for i in range(1, t.shape[0]):
		updatefe(I,V,m,n,h,p,q,r,u,i,tstep,r_mat)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatehines(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u,int i,double tstep,int fineness):
	# currents
	cdef double cV = V[i-1]
	cdef double cm = m[i-1]
	cdef double cn = n[i-1]
	cdef double ch = h[i-1]
	cdef double cp = p[i-1]
	cdef double cq = q[i-1]
	cdef double cr = r[i-1]
	cdef double cu = u[i-1]

	cdef double I_Na, I_K, I_leak, I_M, I_L, I_T, I_ion
	cdef double dt = tstep / fineness

	cdef int j

	cdef double A
	cdef double B
	cdef double Cn, Dn, Cm, Dm, Ch, Dh, Cp, Dp, Cq, Dq, Cr, Dr, Cu, Du

	cdef double dw1, dw2
	cdef double dz1, dz2
	cdef double q1

	cdef np.ndarray[double,ndim=1] rl = np.random.normal(size = 2 * fineness)
	cdef np.ndarray[double,ndim=1] wl = np.random.normal(size = 2 * fineness)
	for j in range(fineness):
		A = -(cm ** 3)*gbar_Na*ch - (cn ** 4) * gbar_K - g_leak - gbar_M * cp
		- gbar_L*(cq**2)*cr - gbar_T*(s_inf(cV)**2)*cu
		B = I[i-1] + (cm ** 3) * gbar_Na * ch * E_Na + (cn**4) * gbar_K * E_K + g_leak * E_leak + gbar_M * cp * E_K
		+ gbar_L*(cq**2)*cr*E_Ca + gbar_T*(s_inf(cV)**2)*cu*E_Ca

		dw1 = wl[2 * j] * sqrt(0.5 * dt)
		dw2 = wl[2 * j + 1] * sqrt(0.5 * dt)
		dz1 = 0.5 * (dt / 2.0) * (dw1 + rl[2 * j] * sqrt(dt / 3.0))
		dz2 = 0.5 * (dt / 2.0) * (dw2 + rl[2 * j + 1] * sqrt(dt / 3.0))

		q1 = dz1 + dz2 - 0.5 * dt * dw1

		cV = (cV + dt * (A * 0.5 * (cV + dw1 - dw2) + B) + nois_fact * (dw1 + dw2) + nois_fact * A * q1) / (1.0 - 0.5 * dt * A)

		Cn = alpha_n(cV)
		Cm = alpha_m(cV)
		Ch = alpha_h(cV)
		Cp = p_inf(cV) / tau_p(cV)
		Cq = alpha_q(cV)
		Cr = alpha_r(cV)
		Cu = u_inf(cV) / tau_u(cV)

		Dn = -(Cn + beta_n(cV))
		Dm = -(Cm + beta_m(cV))
		Dh = -(Ch + beta_h(cV))
		Dp = -1.0 / tau_p(cV)
		Dq = -(Cq + beta_q(cV))
		Dr = -(Cr + beta_r(cV))
		Du = -1.0 / tau_u(cV)

		cn = (cn + dt * (0.5 * Dn * cn + Cn)) / (1.0 - 0.5 * dt * Dn)
		cm = (cm + dt * (0.5 * Dm * cm + Cm)) / (1.0 - 0.5 * dt * Dm)
		ch = (ch + dt * (0.5 * Dh * ch + Ch)) / (1.0 - 0.5 * dt * Dh)
		cp = (cp + dt * (0.5 * Dp * cp + Cp)) / (1.0 - 0.5 * dt * Dp)
		cq = (cq + dt * (0.5 * Dq * cq + Cq)) / (1.0 - 0.5 * dt * Dq)
		cr = (cr + dt * (0.5 * Dr * cr + Cr)) / (1.0 - 0.5 * dt * Dr)
		cu = (cu + dt * (0.5 * Du * cu + Cu)) / (1.0 - 0.5 * dt * Du)

	V[i] = cV
	n[i] = cn
	m[i] = cm
	h[i] = ch
	p[i] = cp
	q[i] = cq
	r[i] = cr
	u[i] = cu

# Leapfrog
def hinesmethod(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,np.ndarray[double,ndim=1] q,np.ndarray[double,ndim=1] r,np.ndarray[double,ndim=1] u,double tstep,int fineness):
	#V[0] = E_leak
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])
	q[0] = alpha_q(V[0])/(alpha_q(V[0])+beta_q(V[0]))
	r[0] = alpha_r(V[0])/(alpha_r(V[0])+beta_r(V[0]))
	u[0] = u_inf(V[0])

	for i in range(1, t.shape[0] - 1):
		updatehines(I,V,m,n,h,p,q,r,u,i,tstep,fineness)

	V[t.shape[0] - 1] = V[t.shape[0] - 2]
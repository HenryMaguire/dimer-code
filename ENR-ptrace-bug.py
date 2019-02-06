



G = basis(1,0)
E = basis(2,1)



site_coherence = OX*XO.dag()

OO_proj = OO*OO.dag()
XO_proj = XO*XO.dag()
OX_proj = OX*OX.dag()

sigma_m1 =  OO*XO.dag()
sigma_m2 =  OO*OX.dag()
sigma_x1 = sigma_m1+sigma_m1.dag()
sigma_x2 = sigma_m2+sigma_m2.dag()

ops = [OO_proj, XO_proj, OX_proj]






def J_minimal(omega, Gamma):
    return Gamma*omega

def rate_up(w, T, gamma):
    n = Occupation(w, T)
    rate = 0.5 * pi * n * J(w, gamma)
    return rate

def rate_down(w, T, gamma):
    n = Occupation(w, T)
    rate = 0.5 * pi * (n + 1. ) * J(w, gamma)
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L

def Occupation(omega, T):
    conversion = 0.695
    n =0.
    beta = 0.
    if T ==0.: # First calculate beta
        n = 0.
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
        if sp.exp(omega*beta)-1 ==0.:
            n = 0.
        else:
            n = float(1./(sp.exp(omega*beta)-1))
    return n

def model_liouv(phonon_ops, optical_op, w_0, Gamma, T_ph, alpha_EM, T_EM):
    phonon_rates = [rate_up(w_0, T_ph, Gamma), rate_down(w_0, T_ph, Gamma)]
    optical_rates = [rate_up(w_0, T_ph, Gamma), rate_down(w_0, T_ph, Gamma)]
    phonon_collapses =  [sqrt(r)*op for r in phonon_rates for op in phonon_ops]
    optical_op = [sqrt(r)*op for r in phonon_rates

def get_H_and_L(bias=100., w_2=5000., V = 100., alpha=100.,
                                 T_EM=6000., T_ph =300.,
                                 alpha_EM=0.1, shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3, exc_diff=0):
    # alpha_1 = alpha+alpha_bias
    # Sets up the parameter dict
    N_1 = N_2 = N
    exc = N+exc_diff
    gap = sqrt(bias**2 +4*(V**2))
    phonon_energy = T_ph*0.695


    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = (w_2*dipole_2)/(w_1*dipole_1)
    T_1, T_2 = T_ph, T_ph # Phonon bath temperature

    Gamma_1 = Gamma_2 = Gamma
    w0_2, w0_1 = w_0, w_0 # underdamped SD parameter omega_0

    H_sub = w_1*XO_proj + w_2*OX_proj + V*(site_coherence+site_coherence.dag())

    # now we include the two modes, but don't even couple them to our electronic system
    #coupling_ops = [sigma_m1.dag()*sigma_m1, sigma_m2.dag()*sigma_m2] # system-RC operators

    I_sub = qeye(H_sub.shape[0])
    I = enr_identity([N,N], N+exc_diff)

    H_S = tensor(H_sub, I)

    atemp = enr_destroy([N,N], N+exc_diff)
    
    a_RC_exc = [tensor(I_sub, aa) for aa in atemp] # annhilation ops in exc restr basis
    phonon_operators = []
    for i in range(len(a_RC_exc)):
        A_i = a_RC_exc[i].dag() + a_RC_exc[i]
        H_Ii = alpha*tensor(coupling_ops[i], I)*A_i
        H_RCi = w_0*a_RC_exc[i].dag()*a_RC_exc[i]
        H_S += H_RCi
        phonon_operators.append(A_i)
    
    return [H_sub, H_S], phonon_operators
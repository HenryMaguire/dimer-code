def dimer_ham_RC_full(w_1, w_2, w_xx, V, mu, Omega_1, Omega_2, kap_1, kap_2, N_1, N_2):
    """
    Input: System splitting, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """
    #OO = basis(4,0)
    #XO = basis(4,1)
    #OX = basis(4,2)
    #XX = basis(4,3)
    #sigma_1 = OX*XX.dag() + OO*XO.dag()
    #sigma_2 = XO*XX.dag() + OO*OX.dag()
    #assert sigma_1*OX == sigma_2*XO
    I_RC_1 = qeye(N_1)
    I_RC_2 = qeye(N_2)
    I_dim = qeye(4)
    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag()
    H_dim += V*(XO*OX.dag() + OX*XO.dag())
    print H_dim
    H_dim = tensor(H_dim, I_RC_1, I_RC_2)
    A_EM = tensor(sigma_1+mu*sigma_2, I_RC_1, I_RC_2)

    A_1 = destroy(N_1).dag()+ destroy(N_1)
    A_2 = destroy(N_2).dag()+ destroy(N_2)

    A_1 = tensor(I_dim, A_1, I_RC_2)
    A_2 = tensor(I_dim, I_RC_1, A_2)

    H_I1 = kap_1*tensor(sigma_1.dag()*sigma_1, I_RC_1, I_RC_2)*A_1
    H_I2 = kap_2*tensor(sigma_2.dag()*sigma_2, I_RC_1, I_RC_2)*A_2

    H_RC1 = tensor(I_dim, Omega_1*destroy(N_1).dag()*destroy(N_1), I_RC_2)
    H_RC2 = tensor(I_dim, I_RC_1, Omega_2*destroy(N_2).dag()*destroy(N_2))

    H_S = H_dim + H_RC1 + H_RC2 + H_I1 + H_I2

    return H_S, A_1, A_2, A_EM

from SES_setup import *
import time
from qutip import build_preconditioner, steadystate

def calculate_steadystate(H, L, fill_factor=500, tol=1e-8, persistent=False, method="iterative-lgmres", maxiter=500):
    calculated = False
    ff = fill_factor
    ss = 0
    while not calculated:
        try:
            M=None
            use_precond=False
            if "iterative" in method:
                ti = time.time()
                M, m_info = build_preconditioner(H[1], [L], fill_factor=fill_factor,return_info=True,
                                        drop_tol=1e-4, use_rcm=True, ILU_MILU='smilu_2', maxiter=maxiter)
                use_precond=True
                print "Building preconditioner took {} seconds".format(time.time()-ti)
                # print m_info['ilu_fill_factor']
            ss, info = steadystate(H[1], [L], method=method, M=M,
                                    use_precond=False,
                                    return_info=True, tol=tol, maxiter=6000)
            print "Steady state took {:0.3f} seconds".format(info['solution_time'])
            return ss, info
        except Exception as err:
            raise Exception
            print "Steadystate failed because {}.".format(err)
            if persistent:
                if "tolerance" in str(err):
                    return 0, 0 # if it's a convergence error, don't bother
                elif "preconditioner" in str(err):
                    ff -= 5
                    time.sleep(5) # Basically just giving the computer a breather
                    print("Trying a smaller fill factor ({})...".format(ff))
                    if ff<10:
                        print("Failed with a lower limit fill factor of 38. Skipping...".format(ff))
                        return 0, 0 # ff is too low, don't bother
                else:
                    raise Exception(str(err))
            else:
                print("Skipping...")
                return 0, 0 # don't bother"""

def calculate_steadystate_bootstrap(H, L_full, chop_threshold=1e-8, fill_factor=500, method="iterative-lgmres", maxiter=1000):
    # return ss, info
    # uses chopped L for the preconditioner, but the full Liouvillian for the krylov subspace bit
    from utils import chop
    try:
        M=None
        use_precond = False
        if "iterative" in method:
            ti = time.time()
            M, m_info = build_preconditioner(H[1], [chop(L_full, threshold=chop_threshold)], fill_factor=500,
                                        return_info=True,drop_tol=1e-4, use_rcm=True, ILU_MILU='smilu_2')
            print m_info
            use_precond = True
            print "Building preconditioner took {} seconds".format(time.time()-ti)
        return steadystate(H[1], [L_full], method=method, M=M,
                                    use_precond=True, 
                                    return_info=True, tol=1e-8)

    except Exception as err:
        print(err)
        return (None, None)


def heat_map_calculator(PARAMS,
                        x_axis_parameters=['w0_1', 'w0_2'],
                        y_axis_parameters=['Gamma_1', 'Gamma_2'],
                        x_values=[100., 200., 300.],
                        y_values=[70., 200., 600.],
                        dir_name='heatmap_oG', fill_factor=47,
                        save_data=True, persistent=False, method='direct',
                        threshold=1e-9):
    info_array = np.zeros(( len(y_values), len(x_values)), dtype=dict)
    ss_array = np.zeros(( len(y_values), len(x_values)), dtype=qt.Qobj)
    for i, y in enumerate(y_values):
        for param_labels in y_axis_parameters:
            PARAMS.update({param_labels : y})
        for j, x in enumerate(x_values):
            # scan over each combination of parameters and update the PARAMS dict
            # for each point on the grid
            for param_labels in x_axis_parameters:
                PARAMS.update({param_labels : x})
            if 'exc' in x_axis_parameters:
                PARAMS.update({'exc': PARAMS['N_2']+PARAMS['N_1']-PARAMS['exc_diff']})
            if 'exc_diff' in y_axis_parameters:
                #print (PARAMS['N_2'], PARAMS['N_1'], PARAMS['exc_diff'])
                PARAMS.update({'exc': PARAMS['N_2']+PARAMS['N_1']-PARAMS['exc_diff']})
            ti = time.time()
            silent = True
            if PARAMS['N_1'] >=8:
                silent = False
            H, L = get_H_and_L(PARAMS,silent=silent, threshold=threshold)
            tf = time.time()
            print "N_1 = {}, N_2 = {}, exc= {}".format(PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc'])
            
            ss, info = calculate_steadystate(H, L, fill_factor=fill_factor,
                                                    persistent=persistent, method=method)
            del H, L
            ss_array[i][j], info_array[i][j] = ss, info
            try:
                ts = info['solution_time']
                print "Build time: {:0.3f} \t | \t Solution time: {:0.3f}".format(tf-ti,
                                                                                  ts)
            except TypeError:
                print "N_1 = {}, N_2 = {}, exc= {} - Calculation skipped...".format(PARAMS['N_1'],
                                                                                      PARAMS['N_2'],
                                                                                      PARAMS['exc'])
            

    # Pass variables through so heatmap_plotter knows what to do
    PARAMS.update({'x_axis_parameters': x_axis_parameters,
                             'y_axis_parameters': y_axis_parameters,
                          'x_values': x_values, 'y_values': y_values})
    if save_data:
        k = 1
        saved = False
        while not saved:
            directory = 'DATA/'+dir_name+'_'+str(k)
            if not os.path.exists(directory):
                os.makedirs(directory)
                save_obj(ss_array, directory+'/ss_array')
                save_obj(PARAMS, directory+'/PARAMS')
                save_obj(info_array, directory+'/info_array')
                saved = True
                print "Files saved at {}".format(directory)
            else:
                k+=1
    return ss_array, info_array

if __name__ == "__main__":
    import sys
    try:
        ff = int(sys.argv[1])
    except Exception as e:
        ff = 100
    
    method = 'eigen'
    #method = 'power'
    #x_values=[6][::-1] # N
    #x_values=[5,6,7,8,9,10,11][::-1] # N
    x_values=[3,4,5][::-1] # N
    
    w_2 = 8000.
    alpha = 50./pi

    N = 3
    pap = alpha_to_pialpha_prop(alpha, w_2)
    wc = 100.
    w_0 = 200.
    Gamma = (w_0**2)/wc
    PARAMS = PARAMS_setup(bias=100., w_2=8000., V = 100., pialpha_prop=pap,
                                    T_EM=6000., T_ph =300.,
                                    alpha_EM=0.1, shift=True,
                                    num_cpus=3, N=N, Gamma=Gamma, w_0=w_0,
                                    silent=True, exc_diff=N)
    exc = PARAMS['exc']

     # N
    #y_values=[5,4,3,2,1,0] # exc
    y_values=[3,2, 1] # exc_diff (exc = 2n-exc_diff)
    """try:
        if 'iterative' in method:
            print("Using fill-factor of {}".format(ff))
    """
    ss_array, info_array= heat_map_calculator(PARAMS,
                            x_axis_parameters=['N_1', 'N_2'],
                            y_axis_parameters=['exc_diff'],
                            x_values=x_values,
                            y_values=y_values,
                            dir_name='heatmap_excvN',
                            fill_factor=ff, save_data=False, 
                            persistent=False, method=method, threshold=1e-7)
    #except Exception as err:
    #    print "Error: This means that an error has been raised internally in the SS calculator. Message {}".format(err)
    #    raise

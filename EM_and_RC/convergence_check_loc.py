##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
#from numpy import matrix
#from numpy import linalg
import numpy as np

from scipy import *

#import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import time as time
#import the RC master script:
import RC_function_steady_state_tester as rcfunc
from multiprocessing import Pool




#what parameters do we want to use?

#tunneling is:
V = 100.
#bias is
eps =50# 50.
print('this is with V={0}cm^-1'.format(V))
#cut-off frequency
wc = 53.

#Temperature:

Temp = 300

w2 = 2000
wXX = 2. * w2 + eps 
n_list = [10]#[11,12]#[4, 6, 8, 10] 

#Number = 8
#excitation = 4
alp = 500. / np.pi

def dat_func(n):
	sol_dat = rcfunc.RCdimerfunction_steadystate_ENR(wXX, w2, eps, 0.5 * V, Temp, wc, 2 * alp, 2 * n, n)
	return sol_dat

#number_pools = 1# None	
#p = Pool(number_pools)	

st = time.time()
sol_three = dat_func(n_list[0])
en = time.time()


print("total time taken ={0}".format(en-st))

#print(sol_three)
#out_dat=array([[n_list[k]]+sol_three for k in range(len(n_list))])
#print(out_dat)

#give the data structure:
dat_struct = 'The data is aranged as : alpha, ground, dark, bright, biexc, coherence'

#write the data to file
#file_name = '/pc2013-data1/woolland/Jake/data_files/convergencedata_largeN.txt'#'/samdata1/woolland/Jake/data_files/steadystatedata_N={0}_excitations={1}.txt'.format(Number,excitation)
#np.savetxt(file_name, out_dat,fmt='%.4e',delimiter = ',', header = dat_struct)


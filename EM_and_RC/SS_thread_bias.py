##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
#from numpy import matrix
#from numpy import linalg
import numpy as np
from multiprocessing import Pool

from scipy import *

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import time as time
#import the RC master script:
import RC_function_steady_state as rcfunc


#what parameters do we want to use?

#tunneling is:
V = 100.
#bias is
eps = 50.
print('this is with V={0}cm^-1'.format(V))
#cut-off frequency
wc = 53.

#Temperature:

Temp = 300.

w2 = 2000
wXX = 2. * w2 + eps 
bias_list = linspace(0., 100., 20) 
Number = 8
excitation = 4	

def data_func(eps):
	sol = rcfunc.RCdimerfunction_steadystate_ENR(wXX, w2, eps, 0.5 * V, Temp, wc, 2 * 2 / pi, Number, excitation)
	return sol
p = Pool(20)

st = time.time()
data = p.map(data_func, bias_list)
en = time.time()


print("time taken to do ENR RCs ={0}".format(en-st))


out_dat=array([[bias_list[k]] + data[k] for k in range(len(bias_list))])
#print(out_dat)

#give the data structure:
dat_struct = 'The data is aranged as : alpha, ground, dark, bright, biexc, coherence'

#write the data to file
file_name ='steadystatedata_bias_N={0}_excitations={1}.txt'.format(Number,excitation)
np.savetxt(file_name, out_dat,fmt='%.4e',delimiter = ',', header = dat_struct)
# grpop = [real(sol_three[k][0]) for k in range(len(alpha_list))]
# darkpop = [real(sol_three[k][1]) for k in range(len(alpha_list))]
# brightpop = [real(sol_three[k][2]) for k in range(len(alpha_list))]
# bipop = [real(sol_three[k][3]) for k in range(len(alpha_list))]
# brightcoh = [sol_three[k][4] for k in range(len(alpha_list))]

# print(grpop,brightcoh)
# 
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(pi * alpha_list, grpop, label = 'Ground',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, darkpop, label = 'Dark',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, brightpop, label = 'Bright',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, bipop, label = 'Biexc', marker = '.', markersize = 10 , linestyle = '')
# #ax.plot(steplist, sol.expect[0], marker = '.', markersize = 10 ,markevery = (0.0,0.04),linestyle = '')
# ax.legend(loc = 0)
# plt.xlabel(r'$pi \alpha$', fontsize = 22)
# plt.ylabel('Eigenstate populations', fontsize = 22)
# plt.show()
# # 
# 
# 
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(pi * alpha_list, real(brightcoh),marker = '.', markersize = 10 , linestyle = '', color = 'b', label = r'$re(\rho_{+-})$')
# ax.plot(pi * alpha_list, imag(brightcoh),marker = '.', markersize = 10 , linestyle = '', color = 'r' ,label = r'$im(\rho_{+-})$')
# #ax.plot(steplist, sol.expect[0], marker = '.', markersize = 10 ,markevery = (0.0,0.04),linestyle = '')
# ax.legend(loc = 0)
# plt.xlabel(r'$pi \alpha$', fontsize = 22)
# plt.ylabel(r'$coherences$', fontsize = 22)
# plt.show()
# 

##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg


from scipy import *

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import time as time
#import the RC master script:
import RC_function_steady_state as rcfunc

import resource


rsrc = resource.RLIMIT_STACK
soft, hard = resource.getrlimit(rsrc)
print(soft, hard)

KILO = 1024

resource.setrlimit(rsrc, (10 ** 3 * KILO, 10 ** 3 * KILO))

soft, hard = resource.getrlimit(rsrc)
print(soft, hard)


#what parameters do we want to use?

#tunneling is:
V = 100.
#bias is
eps =0.# 50.
print('this is with V={0}cm^-1'.format(V))
#cut-off frequency
wc = 53.

#Temperature:

Temp = 300.



#information for the ODE solver
stepsize = 5000
propto = 1.
steplist =  linspace(0, propto, stepsize)

#coupling strength (in terms of the single RC)
# alp = 100. / pi

# st = time.time()
# sol = rcfunc.RCfunction(eps, V, Temp, wc, alp, 10, propto, stepsize)
# en = time.time()
# print("time taken to do single RC ={0}".format(en-st))

w2 = 2000
wXX = 2. * w2 + eps 
alpha_list = [500.0]#linspace(2., 500., 15) / pi


#st = time.time()
sol_three = [rcfunc.RCdimerfunction_steadystate_ENR(wXX, w2, eps, 0.5 * V, Temp, wc, 2 * alp, 10, 5) for alp in alpha_list]
#en = time.time()


#print("time taken to do ENR RCs ={0}".format(en-st))

grpop = [real(sol_three[k][0]) for k in range(len(alpha_list))]
darkpop = [real(sol_three[k][1]) for k in range(len(alpha_list))]
brightpop = [real(sol_three[k][2]) for k in range(len(alpha_list))]
bipop = [real(sol_three[k][3]) for k in range(len(alpha_list))]
brightcoh = [sol_three[k][4] for k in range(len(alpha_list))]
print(sol_three)
#print(grpop,brightcoh)
# 
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(pi * alpha_list, grpop, label = 'Ground',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, darkpop, label = 'Dark',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, brightpop, label = 'Bright',marker = '.', markersize = 10 ,linestyle = '')
# ax.plot(pi * alpha_list, bipop, label = 'Biexc', marker = '.', markersize = 10 , linestyle = ''))
# ax.legend(loc = 0)
# plt.xlabel(r'$pi \alpha$', fontsize = 22)
# plt.ylabel('Eigenstate populations', fontsize = 22)
# plt.show()
# # # 
# # 
# # 
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(pi * alpha_list, real(brightcoh),marker = '.', markersize = 10 , linestyle = '', color = 'b', label = r'$re(\rho_{+-})$')
# ax.plot(pi * alpha_list, imag(brightcoh),marker = '.', markersize = 10 , linestyle = '', color = 'r' ,label = r'$im(\rho_{+-})$')
# ax.legend(loc = 0)
# plt.xlabel(r'$pi \alpha$', fontsize = 22)
# plt.ylabel(r'$coherences$', fontsize = 22)
# plt.show()
# # 

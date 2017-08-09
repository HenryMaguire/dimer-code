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
import RC_function as rcfunc

reload(rcfunc)
#what parameters do we want to use?

#tunneling is:
V = 100.
#bias is
eps = 50.#50.
print('this is with V={0}cm^-1'.format(V))
#cut-off frequency
wc = 53.

#Temperature:

Temp = 300



#information for the ODE solver
stepsize = 5000
propto = 5.
steplist =  linspace(0, propto, stepsize)

#coupling strength (in terms of the single RC)
alp = 4. / pi

# st = time.time()
# sol = rcfunc.RCfunction(eps, V, Temp, wc, alp, 10, propto, stepsize)
# en = time.time()
# print("time taken to do single RC ={0}".format(en-st))

w2 = 1500
wXX = 2. * w2 +eps

st = time.time()
#RCdimerfunction_ENR(wXX, w2, eps, V, Temp, wc, alpha, N, excitations, propto, stepsize)
sol_three = rcfunc.RCdimerfunction_ENR(wXX, w2, eps, V, Temp, wc, alp, 4, 2, propto, stepsize)
en = time.time()
# # #
print("time taken to do ENR RCs ={0}".format(en-st))
# #
# #


#
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(steplist, sol_three.expect[0],linewidth = 2.0, label = 'Ground')
ax.plot(steplist, sol_three.expect[1],linewidth = 2.0, label = 'Dark')
ax.plot(steplist, sol_three.expect[2],linewidth = 2.0, label = 'Bright')
ax.plot(steplist, sol_three.expect[3],linewidth = 2.0, label = 'Biexc')
ax.legend(loc = 0)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Eigenstate populations', fontsize = 22)
plt.savefig('Eig_Pop_alp={0}_time={1}.pdf'.format(pi * alp, propto))
plt.show()
#
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(steplist, real(sol_three.expect[4]),linewidth = 2.0, color = 'b', label = 'Real')#r'$re(\rho_{+-})$')
ax.plot(steplist, imag(sol_three.expect[4]),linewidth = 2.0, color = 'r' ,label = 'Imag')#r'$im(\rho_{+-})$')
ax.legend(loc = 0)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Eigenstate coherences', fontsize = 22)
plt.savefig('Eig_coh_alp={0}_time={1}.pdf'.format(pi * alp, propto))
plt.show()

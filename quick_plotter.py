import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import *
matplotlib.style.use('ggplot')
colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])]
N = 4
fig_name = "5ab"
obs_type = "coherence" # "dark-state"
param_set = "O1"

p_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/p_data".format(param_set,fig_name,N, N))
s_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/s_data".format(param_set,fig_name,N, N))
ns_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/ns_data".format(param_set,fig_name,N, N))
t = np.linspace(0,1,len(s_data.expect[4]))
"""n = 6
if obs_type == "dark_state":
    n = 4"""
"""
n=4
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t, ns_data.expect[n], linestyle='dotted', label='Non-Secular')
ax1.plot(t, s_data.expect[n], linestyle='dashed', label='Secular')
ax1.plot(t, p_data.expect[n], linestyle='solid', label='Phenom.')
ax1.set_ylabel("Dark state population")
plt.xlim(0,1)"""
n=6
fig = plt.figure()
ax2 = fig.add_subplot(111)
lw = 1.7
ax2.plot(t, ns_data.expect[n], linestyle='solid', label='Non-Secular', color=colors[0], linewidth=lw)
ax2.plot(t, ns_data.expect[n].imag, linestyle='dashed', color=colors[0], linewidth=lw)
ax2.plot(t, s_data.expect[n], linestyle='solid', label='Secular', color=colors[1], linewidth=lw)
ax2.plot(t, s_data.expect[n].imag, linestyle='dashed', color=colors[1], linewidth=lw)
ax2.plot(t, p_data.expect[n], linestyle='solid', label='Phenom.', color=colors[2], linewidth=lw)
ax2.plot(t, p_data.expect[n].imag, linestyle='dashed', color=colors[2], linewidth=lw)
ax2.set_ylabel('Coherence')
ax2.set_xlabel('Time')
plt.xlim(0,1)
plt.legend()
plt.savefig("DATA/QHE_{}notes_fig{}/N{}_exc{}/coherence-comparison.pdf".format(param_set,fig_name,N, N))
#plt.ylabel('Dark state population')

#plt.savefig("DATA/QHE_{}notes_fig{}/N{}_exc{}/full-comparison.pdf".format(param_set,fig_name,N, N))
plt.show()

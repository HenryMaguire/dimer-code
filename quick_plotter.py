import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import *
matplotlib.style.use('ggplot')
N = 3
fig_name = "5ab"
obs_type = "coherence" # "dark-state"
param_set = "O2"

p_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/p_data".format(param_set,fig_name,N, N))
s_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/s_data".format(param_set,fig_name,N, N))
ns_data = load_obj("DATA/QHE_{}notes_fig{}/N{}_exc{}/data/ns_data".format(param_set,fig_name,N, N))
t = np.linspace(0,1,len(s_data.expect[4]))
"""n = 6
if obs_type == "dark_state":
    n = 4"""
n=4
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(t, ns_data.expect[n], linestyle='dotted', label='Non-Secular')
ax1.plot(t, s_data.expect[n], linestyle='dashed', label='Secular')
ax1.plot(t, p_data.expect[n], linestyle='solid', label='Phenom.')
ax1.set_ylabel("Dark state population")
plt.xlim(0,1)
n=6
ax2 = fig.add_subplot(212)
ax2.plot(t, ns_data.expect[n], linestyle='dotted', label='Non-Secular')
ax2.plot(t, s_data.expect[n], linestyle='dashed', label='Secular')
ax2.plot(t, p_data.expect[n], linestyle='solid', label='Phenom.')
ax2.set_ylabel('Real coherence')

#plt.ylabel('Dark state population')
ax1.set_xlabel('Time')
plt.xlim(0,1)
plt.legend()
plt.savefig("DATA/QHE_{}notes_fig{}/N{}_exc{}/full-comparison.pdf".format(param_set,fig_name,N, N))
plt.show()

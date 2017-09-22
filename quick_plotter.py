import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import *
matplotlib.style.use('ggplot')
N = 4
p_data = load_obj("DATA/QHE_O1notes_fig4ab/N{}_exc{}/data/p_data".format(N, N))
s_data = load_obj("DATA/QHE_O1notes_fig4ab/N{}_exc{}/data/s_data".format(N, N))
ns_data = load_obj("DATA/QHE_O1notes_fig4ab/N{}_exc{}/data/ns_data".format(N, N))
t = np.linspace(0,1,len(s_data.expect[4]))
plt.plot(t, p_data.expect[4], linestyle='solid', label='Phenom.')
plt.plot(t, s_data.expect[4], linestyle='dotted', label='Secular')
plt.plot(t, ns_data.expect[4], linestyle='dashed', label='Non-Secular')
plt.ylabel('Dark state population')
plt.xlabel('Time')
plt.xlim(0,1)
plt.legend()
plt.savefig("DATA/QHE_O1notes_fig4ab/N{}_exc{}/dark-state-comparison.pdf".format(N, N))
plt.show()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import *
from qutip import Qobj
matplotlib.style.use('ggplot')
colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])]

N = 5
fig_name = "5ab"
obs_type = "coherence" # "dark-state"
param_set = "O1"
alpha_ph = [0.1, 1, 5, 10, 25, 50, 100]
biases = np.linspace(0,300,10)
idx = 3
bias = biases[idx]

ns_points = []
s_points = []
p_points = []

for alpha in alpha_ph:
    alpha_i = int(alpha)
    ns_data = load_obj(
    "DATA/bias_dependence_wRC667_N{}_V100_wc53/nonsecular/steadystate_DMs_alpha{}".format(N,alpha_i))[idx]
    s_data = load_obj(
    "DATA/bias_dependence_wRC667_N{}_V100_wc53/secular/steadystate_DMs_alpha{}".format(N,alpha_i))[idx]
    p_data = load_obj(
    "DATA/bias_dependence_wRC667_N{}_V100_wc53/phenom/steadystate_DMs_alpha{}".format(N,alpha_i))[idx]
    coh = load_obj(
    "DATA/bias_dependence_wRC667_N{}_V100_wc53/operators/eigcoherence_ops".format(N))[idx]
    dark = load_obj(
    "DATA/bias_dependence_wRC667_N{}_V100_wc53/operators/dark_ops".format(N))[idx]
    op = coh
    ns_points.append((ns_data*op).tr().imag)
    s_points.append((s_data*op).tr().imag)
    p_points.append((p_data*op).tr().imag)
lw = 1.8
plt.plot(alpha_ph, ns_points, color=colors[0],label='Non-Secular', linewidth=lw)
plt.plot(alpha_ph, s_points, color=colors[1], label='Secular', linewidth=lw)
plt.plot(alpha_ph, p_points, color=colors[2], label='Phenom.', linewidth=lw)
plt.title("Bias={}".format(int(bias))+r"$cm^{-1}$")
#plt.ylabel("Steady Dark State Population")
plt.ylabel("Steady Coherence Population")
plt.xlabel("Phonon coupling strength")
plt.legend()
plt.savefig("DATA/bias_dependence_wRC667_N{}_V100_wc53/Cohi_alpha_dep_bias{}.pdf".format(N,int(bias)))
plt.show()

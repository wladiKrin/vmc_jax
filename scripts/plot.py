import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sys

data_fn = sys.argv[1]
fn = data_fn.split("/")[-1].split(".hdf")[0]
print(fn)
L = int(fn.split("L=")[1].split("_")[0])
numHidden = int(fn.split("numHidden=")[1])

matplotlib.use('pgf')

fig, ax = plt.subplots(2,1,sharex=True, figsize=(4,5))

exactX = np.loadtxt("exact_simulation/exact_X_L=%d.txt" % (L))
exactZ = np.loadtxt("exact_simulation/exact_Z_L=%d.txt" % (L))

with h5py.File(data_fn, "r") as f:
    times = np.array(f["time_evolution"]["observables"]["times"])
    en = np.array(f["time_evolution"]["observables"]["energy"]["mean"]).ravel()
    en_var = np.array(f["time_evolution"]["observables"]["energy"]["variance"]).ravel()
    X = np.array(f["time_evolution"]["observables"]["X"]["mean"])
    Z = np.array(f["time_evolution"]["observables"]["Z"]["mean"])
    
    meta_times = np.array(f["time_evolution"]["metadata"]["times"])
    tdvp_err = np.array(f["time_evolution"]["metadata"]["tdvp_error"])
    
    param_times = np.array(f["time_evolution"]["network_checkpoints"]["times"])
    params = np.array(f["time_evolution"]["network_checkpoints"]["checkpoints"])



ax[0].plot(exactZ[:,0], np.mean(exactZ[:,1:], axis=1), c="black", label="Exact")
ax[0].plot(times, np.mean(Z, axis=1), c="red", label="NQS")
ax[0].set_ylabel(r"$\frac{1}{N}\sum_l\langle\hat Z_l\rangle$")

ax[1].semilogy(meta_times, tdvp_err)
ax[1].set_ylabel(r"TDVP err")
ax[1].set_xlabel(r'Time $t/\theta_h$')
ax[1].set_ylim((1e-5,1))

ax[0].legend()
ax[0].set_title("%d qubits, RBM with %d hidden units" % (L, numHidden))
plt.tight_layout()
plt.savefig("figures/"+fn+".pdf")

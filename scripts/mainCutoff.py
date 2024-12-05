import os
print(os.environ['CONDA_DEFAULT_ENV'])
print(os.environ['CONDA_PREFIX'])

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
import flax.linen as nn
import jax.numpy as jnp

import numpy as np

import pandas as pd

import time

import jVMC
from jVMC.util import measure
import jVMC.operator as op
from jVMC.stats import SampledObs
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs

import matplotlib.pyplot as plt
import h5py

import tdvp_imp
from nets.RBMCNN import CpxRBMCNN
from jVMC.nets.rbm import CpxRBM
from nets.RBMNoLog import CpxRBMNoLog
from jVMC.nets.initializers import init_fn_args

from sampler.uniformSampler import UniformSampler
from sampler.cutoffSampler  import CutoffSampler

from functools import partial

import json
import sys

def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

inp = None

if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1], 'r') as f:
        inp = json.load(f)
else:
    if mpi.rank == 0:
        print("Error: No input file given.")
        exit()

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

L = inp["L"]
g = inp["trv_field"]
h = 0.0

eps = 1e-0

dt = inp["dt"]  # Initial time step
integratorTol = inp["integratorTol"]  # Adaptive integrator tolerance
tmax = inp["tmax"]  # Final time
numSamples = inp["numSamples"]
num_hidden=inp["num_hidden"]
filter_size=inp["filter_size"]

param_name = "RBMCNN_cutoffSamp_L="+str(L)+"_g="+str(g)+"_num_hidden="+str(num_hidden)+"_filter_size="+str(filter_size)+"_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+"_tmax="+str(tmax)

outp = jVMC.util.OutputManager(inp["data_dir"]+"/output_"+param_name+".hdf5", append=True)

# Set up variational wave function
print("initializing network")
# net = CpxRBMCNN(
#         F=(filter_size,),
#         channels=(num_hidden,),
#         strides=(1,),
#         bias=False, 
#         periodicBoundary=False,
# )

# net = CpxRBM(numHidden = num_hidden)
# psi = jVMC.vqs.NQS(net, logarithmic=True, seed=4321)  # Variational wave function

net = CpxRBMNoLog(numHidden = num_hidden)
psi = jVMC.vqs.NQS(net, logarithmic=False, seed=4321)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L-1):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz(l + 1))))

for l in range(L):
    hamiltonian.add(op.scal_opstr(g, (op.Sx(l), )))
    hamiltonian.add(op.scal_opstr(h, (op.Sz(l),)))

# Set u GS hamiltonian
H_GS = jVMC.operator.BranchFreeOperator()
for l in range(L):
    H_GS.add(op.scal_opstr(-1.0, (op.Sx(l), )))

# Set up observables
observables = {
    "energy": hamiltonian,
    "X": jVMC.operator.BranchFreeOperator(),
}
for l in range(L):
    observables["X"].add(op.scal_opstr(1. / L, (op.Sx(l), )))

# Set up sampler
exactSampler = jVMC.sampler.ExactSampler(psi, L)
psi2Sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L,
                                 numSamples=numSamples, thermalizationSweeps=50)
uniformSampler = UniformSampler(psi, (L,), numSamples=numSamples)
cutoffSampler = CutoffSampler(psi, (L,), random.PRNGKey(4321), eps, updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L,
                                 numSamples=numSamples, thermalizationSweeps=50)
# uniformSampler2 = UniformSampler(psi, (L,), numSamples=numSamples)

# params = np.load("/Users/wladi/Desktop/test.npy")
# psi.set_parameters(params)
params = psi.get_parameters()
print("Number of parameters: ", params.size)

# Set up sampler

######### GS Search ################

# Set up TDVP
# tdvpEquation = tdvp_imp.TDVP({"lhs": exactSampler, "rhs": exactSampler}, rhsPrefactor=1.,
#                                    pinvTol=1e-8, diagonalShift=10, makeReal='real')
# print("starting GS search")
# jVMC.util.ground_state_search(psi, H_GS, tdvpEquation, exactSampler, numSteps=50)
# outp.write_network_checkpoint(0.0, psi.get_parameters())

# print("loading GS data")
# t, weights = outp.get_network_checkpoint(0)
# psi.set_parameters(weights)

#####################################

print("setting up tdvp equation")
tdvpEquation = tdvp_imp.TDVP({"lhs": cutoffSampler, "rhs": cutoffSampler}, rhsPrefactor=1.j)
# tdvpEquation = tdvp_imp.TDVP({"lhs": psi2Sampler, "rhs": psi2Sampler}, rhsPrefactor=1.j)

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)
# stepper = jVMC.util.stepper.Heun(timeStep=dt)

t = 0.
# Measure initial observables
parameters = []
parameters.append(params) 
obs = measure(observables, psi, exactSampler)
data = []
data.append([t, 
    obs["energy"]["mean"][0], 
    obs["energy"]["variance"][0], 
    obs["energy"]["MC_error"][0], 
    obs["X"]["mean"][0],
    obs["X"]["variance"][0], 
    obs["X"]["MC_error"][0], 
    0, 
    0,
    0,
])


print("starting tdvp equation")
while t < tmax:
# while t <= 0:
    tic = time.perf_counter()
    print(">  t = %f\n" % (t))
    print("================================== whole step =============================================")

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, 
                           normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    # print(dp)
    psi.set_parameters(dp)
    t += dt
    tdvpEquation.set_time(t)

    # Measure observables
    obs = measure(observables, psi, exactSampler)
    #
    # # Write some meta info to screen
    print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    print("    Energy = %f +/- %f" % (obs["energy"]["mean"][0], obs["energy"]["MC_error"][0]))
    print("    xPol = %f +/- %f" % (obs["X"]["mean"][0], obs["X"]["MC_error"][0]))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))
    # params = psi.get_parameters()
    # np.save("/Users/wladi/Desktop/test.npy", params)
    #
    data.append([t, 
        obs["energy"]["mean"][0], 
        obs["energy"]["variance"][0], 
        obs["energy"]["MC_error"][0], 
        obs["X"]["mean"][0],
        obs["X"]["variance"][0], 
        obs["X"]["MC_error"][0], 
        tdvpErr, 
        tdvpRes,
        dt,
    ])

    params = psi.get_parameters()
    parameters.append(params) 

    npdata   = np.array(data)
    npparams = np.array(parameters)

    dfTDVP = pd.DataFrame( {
        "time":       npdata[:, 0],
        "energy":     npdata[:, 1],
        "energy_var": npdata[:, 2],
        "energy_MC":  npdata[:, 3],
        "xPol":       npdata[:, 4],
        "xPol_var":   npdata[:, 5],
        "xPol_MC":    npdata[:, 6],
        "tdvpErr":    npdata[:, 7],
        "tdvpRes":    npdata[:, 8],
        "dt":         npdata[:, 9],
    })
    
    # dfTDVP.to_csv(inp["data_dir"]+"/data_"+param_name+".csv", sep=' ')
    #
    # with h5py.File(inp["data_dir"]+"/data_"+param_name+".h5", 'w') as f:
    #     # If single array, save directly
    #     f.create_dataset("time",       data=npdata[:,0])
    #     f.create_dataset("energy",     data=npdata[:,1])
    #     f.create_dataset("energy_var", data=npdata[:,2])
    #     f.create_dataset("energy_MC",  data=npdata[:,3])
    #     f.create_dataset("xPol",       data=npdata[:,4])
    #     f.create_dataset("xPol_var",   data=npdata[:,5])
    #     f.create_dataset("xPol_MC",    data=npdata[:,6])
    #     f.create_dataset("tdvpErr",    data=npdata[:,7])
    #     f.create_dataset("tdvpRes",    data=npdata[:,8])
    #     f.create_dataset("dt",         data=npdata[:,9])
    #         # If list of arrays, create a group and save each array
    #     grp = f.create_group("params")
    #     for i, arr in enumerate(npparams):
    #         grp.create_dataset(f'params_{i}', data=arr)


# params = psi.get_parameters()
# np.save("/Users/wladi/Desktop/test.npy", params)
tic = time.perf_counter()
print(">  t = %f\n" % (t))
print("done")

data = np.array(data)
fig, axs = plt.subplots(2)
#plt.ylim(data[-1,2], 1)

df = pd.read_csv('ref_L='+str(L)+'.csv')
axs[1].plot(df['time'], df['xPol'], color='red')
axs[1].plot(data[:,0], data[:,4])
axs[0].plot(data[:,0], data[:,1])
axs[1].set_xlim(0,tmax+0.1)
plt.savefig("plot.pdf")

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import argparse
import json
import sys
import time
from functools import partial

import flax
import flax.linen as nn
import h5py
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nets.RBMCNN import CpxRBMCNNLog
from nets.RBM import CpxRBMLog, singleParamState

import jVMC
import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
import jVMC.operator as op
from jVMC.nets.initializers import init_fn_args
from jVMC.nets.rbm import CpxRBM
from jVMC.stats import SampledObs
from jVMC.util import measure
from sampler.uniformSampler import UniformSampler
from sampler.cutoffSampler import CutoffSampler

# i = int(os.environ["ENCAP_PROCID"])

inp = None
with open("input.json", 'r') as f:
    inp = json.load(f)

# Physical paramters:
L = inp["system"]["L"]
J = inp["system"]["J"]
g = -1.
h = 0.

tmax = inp["simulation"]["tmax"]
integratorTol = inp["simulation"]["tol"]
dt = inp["simulation"]["dt"]

invCutoff = inp["simulation"]["invCutoff"]
numSamples = inp["simulation"]["num_samples"]
eps = inp["simulation"]["eps"]
sampler_name = inp["simulation"]["sampler"]

# Network parameters
# filter_size = inp["net"]["filtersize"]
numChannels  = inp["net"]["num_channels"]
    
def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

# # Set up variational wave function
# print("initializing network")
# net = CpxRBMCNNLog(
#         F=(filter_size,),
#         channels=(numChannels,),
#         strides=(1,),
#         bias=False, 
#         periodicBoundary=True,
# )

net = CpxRBMLog(numHidden = numChannels, bias = True)
# net = singleParamState(numHidden = 5, bias = True)

psi = jVMC.vqs.NQS(net, logarithmic=True, seed=4321)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(op.scal_opstr(1., (op.Sy(l), )))

for l in range(L-1):
    hamiltonian.add(op.scal_opstr(J, (op.Sz(l), op.Sz(l+1))))

# Set up observables
observables = {
    "energy": hamiltonian,
    "Z": jVMC.operator.BranchFreeOperator(),
    "X": jVMC.operator.BranchFreeOperator(),
    "Y": jVMC.operator.BranchFreeOperator(),
}
for l in range(L):
    observables["Z"].add(op.scal_opstr(1. / L, (op.Sz(l), )))
    observables["X"].add(op.scal_opstr(1. / L, (op.Sx(l), )))
    observables["Y"].add(op.scal_opstr(1. / L, (op.Sx(l), )))

# Set up sampler
exactSampler = jVMC.sampler.ExactSampler(psi, L)

param_name = "RBM_SNRtest_" + sampler_name + "Samp_L="+str(L) + "_J="+str(J)+"_numChannels="+str(numChannels)

if sampler_name == "exact":
    sampler = jVMC.sampler.ExactSampler(psi, L)
elif sampler_name == "psiSq":
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                     numChains=25, sweepSteps=L,
                                     numSamples=numSamples, thermalizationSweeps=25)
elif sampler_name == "cutoff":
    param_name = param_name + "_eps="+str(eps)
    sampler = CutoffSampler(psi, (L,), random.PRNGKey(4321), eps=eps, updateProposer=jVMC.sampler.propose_spin_flip_Z2, numChains=25, numSamples=numSamples, thermalizationSweeps=25, sweepSteps=L)

elif sampler_name == "uniform":
    sampler = UniformSampler(psi, (L,), random.PRNGKey(4321), exactRenorm=False, numSamples=numSamples)
else:
    print("Sampler " + sampler_name + "not defined")
    exit()

print("name: ", param_name)

params = psi.get_parameters()
new_params = np.zeros(params.size) 
new_params[0] = new_params[0]+1
new_params = jnp.array(new_params + 1e-5*np.random.randn(params.size))

psi.set_parameters(new_params)

print("Number of parameters: ", params.size)
print(psi(exactSampler.basis))

print("setting up tdvp equation")
tdvpEquation = jVMC.util.TDVP(sampler, rhsPrefactor=1.j, pinvCutoff=invCutoff)

# Set up stepper
if integratorTol == 0:
    stepper = jVMC.util.stepper.Heun(timeStep=dt)
else:
    stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)

t = 0.
# Measure initial observables
parameters = []
parameters.append(params) 
obs = measure(observables, psi, exactSampler)

data = []
data.append([t, 
    obs["energy"]["mean"][0], 
    obs["energy"]["variance"][0], 
    # obs["energy"]["MC_error"][0], 
    obs["Z"]["mean"][0],
    obs["Z"]["variance"][0], 
    # obs["Z"]["MC_error"][0], 
    obs["Y"]["mean"][0],
    obs["Y"]["variance"][0], 
    # obs["ZZ"]["MC_error"][0], 
    obs["X"]["mean"][0],
    obs["X"]["variance"][0], 
    # obs["X"]["MC_error"][0], 
    0, 0, 0])

print("Z: ", obs["Z"]["mean"][0])
print("X: ", obs["X"]["mean"][0])

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
    # tdvpEquation.set_time(t)

    # Measure observables
    obs = measure(observables, psi, exactSampler)

    # Write some meta info to screen
    print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    print("    Energy = %f +/- %f" % (obs["energy"]["mean"][0], obs["energy"]["MC_error"][0]))
    print("    zPol = %f +/- %f" % (obs["Z"]["mean"][0], obs["Z"]["MC_error"][0]))
    # print("    xPol = %f +/- %f" % (obs["X"]["mean"][0], obs["X"]["MC_error"][0]))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))
    
    print("state: ", psi(exactSampler.basis))

    data.append([t, 
        obs["energy"]["mean"][0], 
        obs["energy"]["variance"][0], 
        # obs["energy"]["MC_error"][0], 
        obs["Z"]["mean"][0],
        obs["Z"]["variance"][0], 
        # obs["Z"]["MC_error"][0], 
        obs["Y"]["mean"][0],
        obs["Y"]["variance"][0], 
        # obs["ZZ"]["MC_error"][0], 
        obs["X"]["mean"][0],
        obs["X"]["variance"][0], 
        # obs["X"]["MC_error"][0], 
        tdvpErr, tdvpRes, dt])

    npdata = np.array(data)
    dfTDVP = pd.DataFrame( {
        "time":      npdata[:, 0],
        "energy":    npdata[:, 1],
        "energy_var":npdata[:, 2],
        "zPol":      npdata[:, 3],
        "zPol_var":  npdata[:, 4],
        "yPol":        npdata[:, 5],
        "yPol_var":    npdata[:, 6],
        "xPol":      npdata[:, 7],
        "xPol_var":  npdata[:, 8],
        "tdvpErr":   npdata[:, 9],
        "tdvpRes":   npdata[:, 10],
        "dt":        npdata[:, 11],
    })

    dfTDVP.to_csv("./data_"+param_name+".csv", sep=' ')

dfPsi = pd.read_csv('./exact/psi_L=10_J=-0.100000.csv', delim_whitespace = True)
psiRef = jnp.array(dfPsi['psiR']) + 1j * jnp.array(dfPsi['psiI'])
psiRef = psiRef[::-1]
psiRef /= jnp.linalg.norm(psiRef)

psiNet = psi(exactSampler.basis)[0,:]
psiNet = jnp.exp(psiNet)
psiNet = psiNet / jnp.linalg.norm(psiNet)


res = jnp.abs(jnp.vdot(psiRef, psiNet))**2
print("overlap: ", res)

npdata = np.array([res])
dfOverlap = pd.DataFrame( {
    "overlap":   np.data
})

dfOverlap.to_csv("./overlap_"+param_name+".csv", sep=' ')

tic = time.perf_counter()
print(">  t = %f\n" % (t))
print("done")

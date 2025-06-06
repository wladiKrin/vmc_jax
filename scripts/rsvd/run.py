import os
import json
import jax

import math

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
from nets.CNN_resnet import ResNet

import jVMC
import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
import jVMC.operator as op
from jVMC.nets.initializers import init_fn_args
from jVMC.nets.rbm import CpxRBM
from jVMC.stats import SampledObs
from jVMC.util import measure

i = int(os.environ["ENCAP_PROCID"])

inp = None
with open("input.json", 'r') as f:
    inp = json.load(f)

# Physical paramters:
L = inp["system"]["L"]
J = -1.
g = -1.
h = 0.

tmax = inp["simulation"]["tmax"]
integratorTol = inp["simulation"]["tol"]
dt = inp["simulation"]["dt"]

invCutoff = inp["simulation"]["invCutoff"]
numSamples = inp["simulation"]["num_samples"]

# rsvd
diagonalizeOnDevice = inp["simulation"]["diagonalizeOnDevice"]
use_rsvd = inp["simulation"]["use_rsvd"]
n_oversamples = inp["simulation"]["n_oversamples"]
n_iter = inp["simulation"]["n_iter"]
n_components_ratio = inp["simulation"]["n_components"][i]

# Network parameters
filter_size = inp["net"]["filtersize"]
numChannels  = inp["net"]["num_channels"]
    
def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

param_name = "ResNet_psi2Samp_L="+str(L)+"_numChannels="+str(numChannels)+"_numSamples="+str(numSamples)+"_rsvd="+str(use_rsvd)


# Set up variational wave function
print("initializing network")
net = CpxRBMCNNLog(
        F=(filter_size,),
        channels=(numChannels,),
        strides=(1,),
        bias=False, 
        periodicBoundary=True,
)

# net = ResNet(
#         F=(filter_size,),
#         channels=(numChannels,numChannels),
#         strides=(1,),
#         bias=True, 
# )

psi = jVMC.vqs.NQS(net, logarithmic=True, seed=4321)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L-1):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz(l + 1))))

for l in range(L):
    hamiltonian.add(op.scal_opstr(g, (op.Sx(l), )))
    hamiltonian.add(op.scal_opstr(h, (op.Sz(l),)))

# Set up observables
observables = {
    "energy": hamiltonian,
    "Z": jVMC.operator.BranchFreeOperator(),
    "ZZ": jVMC.operator.BranchFreeOperator(),
    "X": jVMC.operator.BranchFreeOperator(),
}
for l in range(L):
    observables["Z"].add(op.scal_opstr(1. / L, (op.Sz(l), )))
    observables["X"].add(op.scal_opstr(1. / L, (op.Sx(l), )))
for l in range(L-1):
    observables["ZZ"].add(op.scal_opstr(1. / L, (op.Sz(l), op.Sz(l + 1))))

# Set up sampler
# sampler = jVMC.sampler.ExactSampler(psi, L)
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L,
                                 numSamples=numSamples, thermalizationSweeps=25)

params = psi.get_parameters()
print("Number of parameters: ", params.size)
n_components = math.floor(n_components_ratio * params.size)
if use_rsvd:
    param_name = param_name + "_n_components="+str(n_components) 

######### GS Search ################
# Set u GS hamiltonian
# H_GS = jVMC.operator.BranchFreeOperator()
# for l in range(L):
#     H_GS.add(op.scal_opstr(-1.0, (op.Sx(l), )))

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
tdvpEquation = jVMC.util.TDVP(sampler, rhsPrefactor=1.j, 
        diagonalizeOnDevice=diagonalizeOnDevice, 
        randomSVD=use_rsvd,
        n_components=n_components,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
)

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)

t = 0.
# Measure initial observables
parameters = []
parameters.append(params) 
obs = measure(observables, psi, sampler)
data = []
data.append([t, 
    obs["energy"]["mean"][0], 
    obs["energy"]["variance"][0], 
    # obs["energy"]["MC_error"][0], 
    obs["Z"]["mean"][0],
    obs["Z"]["variance"][0], 
    # obs["Z"]["MC_error"][0], 
    obs["ZZ"]["mean"][0],
    obs["ZZ"]["variance"][0], 
    # obs["ZZ"]["MC_error"][0], 
    obs["X"]["mean"][0],
    obs["X"]["variance"][0], 
    # obs["X"]["MC_error"][0], 
    0, 0, 0, 0])


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
    obs = measure(observables, psi, sampler)

    # Write some meta info to screen
    print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    print("    Energy = %f +/- %f" % (obs["energy"]["mean"][0], obs["energy"]["MC_error"][0]))
    print("    xPol = %f +/- %f" % (obs["X"]["mean"][0], obs["X"]["MC_error"][0]))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))

    data.append([t, 
        obs["energy"]["mean"][0], 
        obs["energy"]["variance"][0], 
        # obs["energy"]["MC_error"][0], 
        obs["Z"]["mean"][0],
        obs["Z"]["variance"][0], 
        # obs["Z"]["MC_error"][0], 
        obs["ZZ"]["mean"][0],
        obs["ZZ"]["variance"][0], 
        # obs["ZZ"]["MC_error"][0], 
        obs["X"]["mean"][0],
        obs["X"]["variance"][0], 
        # obs["X"]["MC_error"][0], 
        tdvpErr, tdvpRes, dt, toc-tic])

    npdata = np.array(data)
    dfTDVP = pd.DataFrame( {
        "time":      npdata[:, 0],
        "energy":    npdata[:, 1],
        "energy_var":npdata[:, 2],
        "zPol":      npdata[:, 3],
        "zPol_var":  npdata[:, 4],
        "zz":        npdata[:, 5],
        "zz_var":    npdata[:, 6],
        "xPol":      npdata[:, 7],
        "xPol_var":  npdata[:, 8],
        "tdvpErr":   npdata[:, 9],
        "tdvpRes":   npdata[:, 10],
        "dt":        npdata[:, 11],
        "sim_time":  npdata[:, 12],
    })

    dfTDVP.to_csv("./data_"+param_name+".csv", sep=' ')

tic = time.perf_counter()
print(">  t = %f\n" % (t))
print("done")

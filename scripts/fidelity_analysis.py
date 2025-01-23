import os
import os.path

import jax
jax.config.update("jax_enable_x64", True)

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
from sampler.uniformSampler import UniformSampler
from nets.RBMCNN import CpxRBMCNN, CpxRBMCNNLog
from nets.CNN_resnet import ResNet
from jVMC.nets.initializers import init_fn_args

from functools import partial

import json
import sys

L = 20
g = -1.0
h = 0.0

numSamples    = 40000

num_hidden    = 20
filter_size   = 10

tmax          = 2.0
dt            = 1e-4
integratorTol = 0.0001 # 1e-4
invCutoff     = 1e-8

def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

print("initializing network")
# param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)
param_name = "RBMCNNLog_exactSamp_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_exactRenorm=True_integratorTol="+str(integratorTol)+"_invCutoff="+str(invCutoff)+ "_tmax="+str(tmax)

net = CpxRBMCNNLog(
        F=(filter_size,),
        channels=(num_hidden,),
        strides=(1,),
        bias=False, 
        periodicBoundary=False,
)

psi = jVMC.vqs.NQS(net, logarithmic=True, seed=4321)  # Variational wave function
# param_name = "RBMCNNLog_exactSamp_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+ "_exactRenorm="+str(exactRenorm) +"_integratorTol="+str(integratorTol)+ "_invCutoff="+str(invCutoff)+ "_tmax="+str(tmax)

# Set up variational wave function
# net = ResNet(
#         F=(filter_size,),
#         channels=(num_hidden,),
#         strides=(1,),
#         bias=False, 
# )
# 
# psi = jVMC.vqs.NQS( 
#         net, 
#         logarithmic=True, 
#         seed=4321,
# )  # Variational wave function

exactSampler = jVMC.sampler.ExactSampler(psi, L)

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L-1):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz(l + 1))))

for l in range(L):
    hamiltonian.add(op.scal_opstr(g, (op.Sx(l), )))
    hamiltonian.add(op.scal_opstr(h, (op.Sz(l),)))


ref_name = "wvfct_nqs_exact_L=20.h5"
params = psi.get_parameters()
print("Number of parameters: ", params.size)

ts = []
data = []

with h5py.File("../data/data_"+param_name+".h5", 'r') as f:
    timesRead = f['time'][:]
    print(timesRead)
    with h5py.File("../data/"+ref_name, 'r') as fR:
        print(fR['J=-1_g=-1.0'].keys())


        for (i,t) in enumerate(timesRead):
            print("time: ", t)

            psiRef = fR['J=-1_g=-1.0']['t='+str(round(t,4))][:]
            psiRef = psiRef / jnp.linalg.norm(psiRef)

            # print(psiRef.shape)
                    
            params = f['params']['params_'+str(i)][:]
            exactSampler = jVMC.sampler.ExactSampler(psi, L)
            psi.set_parameters(params)
            psiNet = psi(exactSampler.basis)[0,:]
            psiNet = jnp.exp(psiNet)
            psiNet = psiNet / jnp.linalg.norm(psiNet)

            res = jnp.abs(jnp.vdot(psiRef, psiNet))**2
            print(res)
            ts.append(round(t,4))
            data.append(res)

            npts     = np.array(ts)
            npdata   = np.array(data)

            dfTDVP = pd.DataFrame( {
                "time":       npts,
                "fidelity":   npdata,
            })

            dfTDVP.to_csv("../data/fidelity_"+param_name+".csv", sep=' ')

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
from jVMC.nets.rbm import CpxRBM
from nets.RBMNoLog import CpxRBMNoLog
from jVMC.nets.initializers import init_fn_args

from functools import partial

import json
import sys

L = 20
g = -1.0
h = 0.0

numSamples    = 100000.0

num_hidden    = 20
filter_size   = 10

tmax          = 2.0
dt            = 1e-4
integratorTol = 0.0001 # 1e-4
invCutoff     = 1e-10

def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)
# param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_invCutoff="+str(invCutoff)+ "_tmax="+str(tmax)
# param_name = "RBMCNN_mixedSamp_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)

# Set up variational wave function
print("initializing network")
net = CpxRBMCNN(
        F=(filter_size,),
        channels=(num_hidden,),
        strides=(1,),
        bias=False, 
        periodicBoundary=False,
)

psi = jVMC.vqs.NQS(net, logarithmic=False, seed=4321)  # Variational wave function
initSampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=5, sweepSteps=L,
                                 numSamples=10, thermalizationSweeps=5)

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L-1):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz(l + 1))))

for l in range(L):
    hamiltonian.add(op.scal_opstr(g, (op.Sx(l), )))
    hamiltonian.add(op.scal_opstr(h, (op.Sz(l),)))


numSampless = np.array([8,16,32,64])*1000

params = psi.get_parameters()
print("Number of parameters: ", params.size)

# meansW = []
# meansU = []
# meansM = []
# 
# mediansW = []
# mediansU = []
# mediansM = []
# 
# varsW = []
# varsU = []
# varsM = []

with h5py.File("../data/data_"+param_name+".h5", 'r') as f:
    # If single array, save directly
    timesRead = f['time'][:]

    print(timesRead)

    file_name = "../data/SR_analysis_"+param_name+".h5"
    with h5py.File(file_name, 'w') as fW:
        for numSamples in numSampless:
            numT = 0
            timesSave = 0

            print("numSamples: ", numSamples)
            print(fW.keys())

            grpN = fW.create_group(f'numSamples_{numSamples}')

            for (i,t) in enumerate(timesRead):
                if t >= timesSave:
                    print("time: " + str(t))
                    params = f['params']['params_'+str(i)][:]
                    
                    psi.set_parameters(params)

                    psi2Sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                                     numChains=25, sweepSteps=L,
                                                     numSamples=numSamples, thermalizationSweeps=25)
                    uniformSampler = UniformSampler(psi, (L,), numSamples=numSamples, exactRenorm=False)
                    exactSampler = jVMC.sampler.ExactSampler(psi, (L,))

                    grp = grpN.create_group(f'step_{numT}')

                    grp.create_dataset("time", data=t)

                    grpS = grp.create_group("S_matrix")
                    grpF = grp.create_group("F_vector")

                    print("starting exact")

                    sampleEconfigs, sampleEcoeffs, sampleEweights = exactSampler.sample(numSamples=numSamples)

                    ElocE = SampledObs(hamiltonian.get_O_loc(sampleEconfigs, psi, sampleEcoeffs, t), sampleEweights)
                    ElocEMean = ElocE.mean()[0]
                    elocE = jnp.sqrt(sampleEweights[:,:,None])*(sampleEcoeffs[:,:,None]*ElocE.obs[:,:,None] - sampleEcoeffs[:,:,None]*ElocEMean)

                    gradientsE = psi.gradients(sampleEconfigs)
                    print("normGrad: ", np.linalg.norm(gradientsE))
                    gradsEMean = jnp.sum(sampleEweights[:,:,None] * gradientsE * jnp.conj(sampleEcoeffs[:,:,None]), axis = 1)
                    gradsE = jnp.sqrt(sampleEweights[:,:,None])*(gradientsE - sampleEcoeffs[:,:,None]*gradsEMean[None,:,:])

                    SE = mpi.global_sum(jVMC.stats._covar_helper(gradsE, gradsE)[:,None,...])
                    FE = mpi.global_sum(jVMC.stats._covar_helper(gradsE, elocE)[:,None,...]).ravel()
                    
                    print("normS: ", np.linalg.norm(SE))

                    grpS.create_dataset("exact", data=SE)
                    grpF.create_dataset("exact", data=FE)

                    # print("exact done")

                    grpSU = grpS.create_group("uniform")
                    grpSW = grpS.create_group("wvsq")
                    grpSM = grpS.create_group("mixed")

                    grpFU = grpF.create_group("uniform")
                    grpFW = grpF.create_group("wvsq")
                    grpFM = grpF.create_group("mixed")

                    # meanSW = 0
                    # meanSU = 0
                    # meanSM = 0

                    # medianSW = 0
                    # medianSU = 0
                    # medianSM = 0

                    # varSW = 0
                    # varSU = 0
                    # varSM = 0

                    # meanFW = 0
                    # meanFU = 0
                    # meanFM = 0

                    # medianFW = 0
                    # medianFU = 0
                    # medianFM = 0

                    # varFW = 0
                    # varFU = 0
                    # varFM = 0

                    for num in np.arange(20): #20
                        print(f'num = {num}')
                        sampleUconfigs, sampleUcoeffs, sampleUweights = uniformSampler.sample(numSamples=numSamples)
                        sampleWconfigs, sampleWcoeffs, sampleWweights = psi2Sampler.sample(numSamples=numSamples)

                        ElocU = SampledObs(hamiltonian.get_O_loc(sampleUconfigs, psi, sampleUcoeffs, t), sampleUweights)
                        ElocW = SampledObs(hamiltonian.get_O_loc(sampleWconfigs, psi, sampleWcoeffs, t), sampleWweights)

                        ElocUMean = ElocU.mean()[0]
                        ElocWMean = ElocW.mean()[0]

                        gradientsU = psi.gradients(sampleUconfigs)
                        gradientsW = psi.gradients(sampleWconfigs)

                        gradsUMean = jnp.sum(sampleUweights[:,:,None] * gradientsU * jnp.conj(sampleUcoeffs[:,:,None]), axis = 1)
                        gradsWMean = jnp.sum(sampleWweights[:,:,None] * gradientsW * jnp.conj(sampleWcoeffs[:,:,None]), axis = 1)

                        gradsU = jnp.sqrt(sampleUweights[:,:,None])*(gradientsU - sampleUcoeffs[:,:,None]*gradsUMean[None,:,:])
                        gradsW = jnp.sqrt(sampleWweights[:,:,None])*(gradientsW - sampleWcoeffs[:,:,None]*gradsWMean[None,:,:])
                        gradsM = jnp.sqrt(sampleUweights[:,:,None])*(gradientsU - sampleUcoeffs[:,:,None]*gradsWMean[None,:,:])

                        elocU = jnp.sqrt(sampleUweights[:,:,None])*(sampleUcoeffs[:,:,None]*ElocU.obs[:,:,None] - sampleUcoeffs[:,:,None]*ElocUMean)
                        elocW = jnp.sqrt(sampleWweights[:,:,None])*(sampleWcoeffs[:,:,None]*ElocW.obs[:,:,None] - sampleWcoeffs[:,:,None]*ElocWMean)
                        elocM = jnp.sqrt(sampleUweights[:,:,None])*(sampleUcoeffs[:,:,None]*ElocU.obs[:,:,None] - sampleUcoeffs[:,:,None]*ElocWMean)

                        SU = mpi.global_sum(jVMC.stats._covar_helper(gradsU, gradsU)[:,None,...])
                        SW = mpi.global_sum(jVMC.stats._covar_helper(gradsW, gradsW)[:,None,...])
                        SM = mpi.global_sum(jVMC.stats._covar_helper(gradsM, gradsM)[:,None,...])

                        # print(np.linalg.norm(SU))
                        # print(np.linalg.norm(SW))
                        # print(np.linalg.norm(SM))

                        FU = mpi.global_sum(jVMC.stats._covar_helper(gradsU, elocU)[:,None,...]).ravel()
                        FW = mpi.global_sum(jVMC.stats._covar_helper(gradsW, elocW)[:,None,...]).ravel()
                        FM = mpi.global_sum(jVMC.stats._covar_helper(gradsM, elocM)[:,None,...]).ravel()

                        grpSU.create_dataset(f'n_{num}', data=SU)
                        grpSW.create_dataset(f'n_{num}', data=SW)
                        grpSM.create_dataset(f'n_{num}', data=SM)

                        grpFU.create_dataset(f'n_{num}', data=FU)
                        grpFW.create_dataset(f'n_{num}', data=FW)
                        grpFM.create_dataset(f'n_{num}', data=FM)


                    timesSave += 0.1
                    numT += 1

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
numSampless = np.array([8,16,32,64])*1000

rhsPrefactor=1.j
def imagFun(x):
    return 0.5 * (x - jnp.conj(x))

param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)

file_name = "../data/SR_analysis_"+param_name+".h5"
with h5py.File(file_name, 'r') as f:
    
    with h5py.File("../data/data_SR_update_"+param_name+".h5", 'w') as fW:

        for numSamples in numSampless:
            times = []

            updateW = []
            updateU = []
            updateM = []

            grpWN = fW.create_group(f'numSamples_{numSamples}')

            updateW = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)
            updateU = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)
            updateM = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)

            print("samp = ", numSamples)
            grpN = f[f'numSamples_{numSamples}']

            for numT in range(len(grpN.keys())):
                print("t = ", numT)

                grpW = grpWN.create_group(f'step_{numT}')
                grp = grpN[f'step_{numT}']
                times.append(grp["time"][()])

                grpS = grp["S_matrix"]
                grpF = grp["F_vector"]

                SE = imagFun(np.array(grpS["exact"][()]))
                FE = imagFun(rhsPrefactor * np.array(grpF["exact"][()]))

                ev, V = jnp.linalg.eigh(SE)
                VtF = jnp.dot(jnp.transpose(jnp.conj(V)), FE)
                # Discard eigenvalues below numerical precision
                
                invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

                residual = 1.0
                cutoff = 1e-2
                pinvTol=1e-14
                pinvCutoff=1e-8

                F_norm = jnp.linalg.norm(FE)
                while residual > pinvTol and cutoff > pinvCutoff:
                    cutoff *= 0.8
                    regularizer = 1. / (1. + (max(cutoff, pinvCutoff) / jnp.abs(ev / ev[-1]))**6)
                    pinvEv = invEv * regularizer
                    residual = jnp.linalg.norm((pinvEv * ev - jnp.ones_like(pinvEv)) * VtF) / F_norm

                updateE = jnp.real(jnp.dot(V, (pinvEv * VtF)))

                grpSU = grpS["uniform"]
                grpSW = grpS["wvsq"]
                grpSM = grpS["mixed"]

                grpFU = grpF["uniform"]
                grpFW = grpF["wvsq"]
                grpFM = grpF["mixed"]

                for num in np.arange(20): #20
                    print(f'num = {num}')

                    SU = imagFun(grpSU[f'n_{num}'][()])
                    SW = imagFun(grpSW[f'n_{num}'][()])
                    SM = imagFun(grpSM[f'n_{num}'][()])
                              
                    FU = imagFun(rhsPrefactor * grpFU[f'n_{num}'][()])
                    FW = imagFun(rhsPrefactor * grpFW[f'n_{num}'][()])
                    FM = imagFun(rhsPrefactor * grpFM[f'n_{num}'][()])



                    ev, V = jnp.linalg.eigh(SU)
                    VtF = jnp.dot(jnp.transpose(jnp.conj(V)), FU)
                    # Discard eigenvalues below numerical precision
                    
                    invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

                    residual = 1.0
                    cutoff = 1e-2
                    pinvTol=1e-14
                    pinvCutoff=1e-8

                    F_norm = jnp.linalg.norm(FU)
                    while residual > pinvTol and cutoff > pinvCutoff:
                        cutoff *= 0.8
                        regularizer = 1. / (1. + (max(cutoff, pinvCutoff) / jnp.abs(ev / ev[-1]))**6)
                        pinvEv = invEv * regularizer
                        residual = jnp.linalg.norm((pinvEv * ev - jnp.ones_like(pinvEv)) * VtF) / F_norm

                    updateU = updateU + jnp.real(jnp.dot(V, (pinvEv * VtF)))



                    ev, V = jnp.linalg.eigh(SW)
                    VtF = jnp.dot(jnp.transpose(jnp.conj(V)), FW)
                    # Discard eigenvalues below numerical precision
                    
                    invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

                    residual = 1.0
                    cutoff = 1e-2
                    pinvTol=1e-14
                    pinvCutoff=1e-8

                    F_norm = jnp.linalg.norm(FW)
                    while residual > pinvTol and cutoff > pinvCutoff:
                        cutoff *= 0.8
                        regularizer = 1. / (1. + (max(cutoff, pinvCutoff) / jnp.abs(ev / ev[-1]))**6)
                        pinvEv = invEv * regularizer
                        residual = jnp.linalg.norm((pinvEv * ev - jnp.ones_like(pinvEv)) * VtF) / F_norm

                    updateW = updateW + jnp.real(jnp.dot(V, (pinvEv * VtF)))



                    ev, V = jnp.linalg.eigh(SM)
                    VtF = jnp.dot(jnp.transpose(jnp.conj(V)), FM)
                    # Discard eigenvalues below numerical precision
                    
                    invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

                    residual = 1.0
                    cutoff = 1e-2
                    pinvTol=1e-14
                    pinvCutoff=1e-8

                    F_norm = jnp.linalg.norm(FM)
                    while residual > pinvTol and cutoff > pinvCutoff:
                        cutoff *= 0.8
                        regularizer = 1. / (1. + (max(cutoff, pinvCutoff) / jnp.abs(ev / ev[-1]))**6)
                        pinvEv = invEv * regularizer
                        residual = jnp.linalg.norm((pinvEv * ev - jnp.ones_like(pinvEv)) * VtF) / F_norm

                    updateM = updateM + jnp.real(jnp.dot(V, (pinvEv * VtF)))


                updateU = updateU / 20
                updateW = updateW / 20
                updateM = updateM / 20

                grpW.create_dataset("time", data=times[-1])
                grpW.create_dataset("exact", data=updateE)
                grpW.create_dataset("uniform", data=updateU)
                grpW.create_dataset("mixed", data=updateM)
                grpW.create_dataset("wvsq", data=updateW)

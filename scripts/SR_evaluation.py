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

param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)
# param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_invCutoff="+str(invCutoff)+ "_tmax="+str(tmax)
# param_name = "RBMCNN_mixedSamp_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)


file_name = "../data/SR_analysis_"+param_name+".h5"
with h5py.File(file_name, 'r') as f:
    

    for numSamples in numSampless:
        times = []

        meansSW = []
        meansSU = []
        meansSM = []
        mediansSW = []
        mediansSU = []
        mediansSM = []
        varsSW = []
        varsSU = []
        varsSM = []

        meansFW = []
        meansFU = []
        meansFM = []
        mediansFW = []
        mediansFU = []
        mediansFM = []
        varsFW = []
        varsFU = []
        varsFM = []

        print("samp = ", numSamples)
        grpN = f[f'numSamples_{numSamples}']
        for numT in range(len(grpN.keys())):
            print("t = ", numT)

            grp = grpN[f'step_{numT}']
            times.append(grp["time"][()])

            grpS = grp["S_matrix"]
            grpF = grp["F_vector"]

            SE = np.array(grpS["exact"][()])
            FE = np.array(grpF["exact"][()])

            grpSU = grpS["uniform"]
            grpSW = grpS["wvsq"]
            grpSM = grpS["mixed"]

            grpFU = grpF["uniform"]
            grpFW = grpF["wvsq"]
            grpFM = grpF["mixed"]

            SU = np.zeros((2*(1+filter_size)*L,2*(1+filter_size)*L), dtype=np.complex128)
            SW = np.zeros((2*(1+filter_size)*L,2*(1+filter_size)*L), dtype=np.complex128)
            SM = np.zeros((2*(1+filter_size)*L,2*(1+filter_size)*L), dtype=np.complex128)

            FU = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)
            FW = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)
            FM = np.zeros(2*(1+filter_size)*L, dtype=np.complex128)

            one = np.ones((2*(1+filter_size)*L,2*(1+filter_size)*L), dtype=np.complex128)
            oneF = np.ones(2*(1+filter_size)*L, dtype=np.complex128)

            for num in np.arange(20): #20
                print(f'num = {num}')
                print(np.linalg.norm(grpSU[f'n_{num}'][()]))

                # SU += np.abs((grpSU[f'n_{num}'][()] - SE)/SE)
                # SW += np.abs((grpSW[f'n_{num}'][()] - SE)/SE)
                # SM += np.abs((grpSM[f'n_{num}'][()] - SE)/SE)

                # FU += np.abs((grpFU[f'n_{num}'][()] - FE)/FE)
                # FW += np.abs((grpFW[f'n_{num}'][()] - FE)/FE)
                # FM += np.abs((grpFM[f'n_{num}'][()] - FE)/FE)

                SU = SU + grpSU[f'n_{num}'][()]
                SW = SW + grpSW[f'n_{num}'][()]
                SM = SM + grpSM[f'n_{num}'][()]
                          
                FU = FU + grpFU[f'n_{num}'][()]
                FW = FW + grpFW[f'n_{num}'][()]
                FM = FM + grpFM[f'n_{num}'][()]


            SU = SU / 20
            SW = SW / 20
            SM = SM / 20
            FU = FU / 20
            FW = FW / 20
            FM = FM / 20

            #print("SE: ", np.linalg.norm(SE))
            # print("SU: ", np.linalg.norm(SU))

            # meansSU.append(np.mean(SU.flatten()))    
            # meansSW.append(np.mean(SW.flatten()))    
            # meansSM.append(np.mean(SM.flatten()))    
            # mediansSU.append(np.median(SU.flatten()))    
            # mediansSW.append(np.median(SW.flatten()))    
            # mediansSM.append(np.median(SM.flatten()))    
            # varsSU.append(np.var(SU.flatten()))    
            # varsSW.append(np.var(SW.flatten()))    
            # varsSM.append(np.var(SM.flatten()))    

            # meansFU.append(np.mean(FU.flatten()))    
            # meansFW.append(np.mean(FW.flatten()))    
            # meansFM.append(np.mean(FM.flatten()))    
            # mediansFU.append(np.median(FU.flatten()))    
            # mediansFW.append(np.median(FW.flatten()))    
            # mediansFM.append(np.median(FM.flatten()))    
            # varsFU.append(np.var(FU.flatten()))    
            # varsFW.append(np.var(FW.flatten()))    
            # varsFM.append(np.var(FM.flatten()))    

            # meansSU.append(np.mean(np.abs(np.divide(SU - SE,SE)).flatten()))    
            
            # print("SU-SE: ", np.abs(SU - SE))
            # print("SE: ", np.abs(SE))
            # print("shape: ", np.abs(SU - SE).flatten().shape)
            # print("divide: ", np.divide(np.abs(SU - SE), np.abs(SE)))
            # print("/: ", np.abs(SU - SE)/ np.abs(SE))

            meansSU.append(np.mean(np.abs(SU - SE).flatten()))    
            meansSW.append(np.mean(np.abs((SW - SE)/SE).flatten()))    
            meansSM.append(np.mean(np.abs((SM - SE)/SE).flatten()))    
            mediansSU.append(np.median(np.abs((SU - SE)/SE).flatten()))    
            mediansSW.append(np.median(np.abs((SW - SE)/SE).flatten()))    
            mediansSM.append(np.median(np.abs((SM - SE)/SE).flatten()))    
            varsSU.append(np.var(np.abs((SU - SE)/SE).flatten()))    
            varsSW.append(np.var(np.abs((SW - SE)/SE).flatten()))    
            varsSM.append(np.var(np.abs((SM - SE)/SE).flatten()))    

            meansFU.append(np.mean(np.abs((FU - FE)/FE).flatten()))    
            meansFW.append(np.mean(np.abs((FW - FE)/FE).flatten()))    
            meansFM.append(np.mean(np.abs((FM - FE)/FE).flatten()))    
            mediansFU.append(np.median(np.abs((FU - FE)/FE).flatten()))    
            mediansFW.append(np.median(np.abs((FW - FE)/FE).flatten()))    
            mediansFM.append(np.median(np.abs((FM - FE)/FE).flatten()))    
            varsFU.append(np.var(np.abs((FU - FE)/FE).flatten()))    
            varsFW.append(np.var(np.abs((FW - FE)/FE).flatten()))    
            varsFM.append(np.var(np.abs((FM - FE)/FE).flatten()))    

            print(meansSU[-1])


            df = pd.DataFrame( {
                "times":     times,

                "meansSU":   meansSU,
                "meansSW":   meansSW,
                "meansSM":   meansSM,
                "mediansSU": mediansSU,
                "mediansSW": mediansSW,
                "mediansSM": mediansSM,
                "varsSU ":   varsSU,
                "varsSW ":   varsSW,
                "varsSM ":   varsSM,
                                      
                "meansFU":   meansFU,
                "meansFW":   meansFW,
                "meansFM":   meansFM,
                "mediansFU": mediansFU,
                "mediansFW": mediansFW,
                "mediansFM": mediansFM,
                "varsFU ":   varsFU,
                "varsFW ":   varsFW,
                "varsFM ":   varsFM,
            })

            param_name = "RBMCNN_mixedSamp_withRenorm_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+ "_tmax="+str(tmax)
            df.to_csv("../data/data_SR_eval_"+param_name+".csv", sep=' ')

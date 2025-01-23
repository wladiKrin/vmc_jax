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
import tdvp_imp
from nets.CNN_resnet import ResNet
#from jVMC.nets.rbm import CpxRBM
from nets.RBMCNN import CpxRBMCNNLog
from sampler.uniformSampler import UniformSampler

import jVMC
import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
import jVMC.operator as op
from jVMC.nets.initializers import init_fn_args
from jVMC.stats import SampledObs
from jVMC.util import measure

# Create the argument parser
# parser = argparse.ArgumentParser( description='TDVP script')

# Positional arguments
# parser.add_argument('-m', '--modelname', 
#     help='modelname for saving purposes',
# )
    
# parser.add_argument('-l', '--lattice', type=int, 
#                     default=10, 
#                     help='lattice size (default: 10)')
# parser.add_argument('-g', '--trvField', type=float, 
#                     default=-1, 
#                     help='transverse field (default: -1)')
#
# parser.add_argument('-s', '--numSamples', type=int, 
#                     default=10000, 
#                     help='Number of samples (default: 1e4)')
# parser.add_argument('--exactRenorm', type=bool, 
#                     default=False, 
#                     help='Wether to use the exact Renormalisation factor (default: false)')
#
# parser.add_argument('--numHidden', type=int, 
#                     default=20, 
#                     help='Number of hidden units (default: 20)')
# parser.add_argument('-f', '--filterSize', type=int, 
#                     default=10, 
#                     help='Filter size (default: 10)')
#
# parser.add_argument('--tmax', type=float, 
#                     default=2., 
#                     help='maximum time (default: 2)')
# parser.add_argument('--dt', type=float, 
#                     default=1e-4, 
#                     help='Time step (default: 1e-4)')
# parser.add_argument('--integratorTol', type=float, 
#                     default=1e-4, 
#                     help='Adaptive Heun integrator tolerance (default: 1e-4)')
#
# parser.add_argument('--invCutoff', type=float, 
#                     default=1e-8, 
#                     help='Cutoff for matrix inversion (default: 1e-8)')
#     
# # Parse the arguments
# args = parser.parse_args()

L = 3
g = -1
h = 0.0

numSamples    = 2000
exactRenorm   = False

num_hidden    = 3
filter_size   = 2

tmax          = 0.5
dt            = 1e-4
integratorTol = 1e-4
invCutoff     = 1e-8

def xy_to_id(x,y,L):
    return int(x + L * y)

def norm_fun(v, df=lambda x: x):
    return jnp.abs(jnp.real(jnp.vdot(v,df(v))))

if mpi.commSize > 1:
    global_defs.set_pmap_devices(jax.devices()[mpi.rank % jax.device_count()])
else:
    global_defs.set_pmap_devices(jax.devices()[0])

print(" -> Rank %d working with device %s" % (mpi.rank, global_defs.devices()), flush=True)

param_name = "ResNetLog_exactSamp_L="+str(L)+ "_g="+str(g)+ "_num_hidden="+str(num_hidden)+ "_filter_size="+str(filter_size)+ "_numSamples="+str(numSamples)+ "_exactRenorm="+str(exactRenorm) +"_integratorTol="+str(integratorTol)+ "_invCutoff="+str(invCutoff)+ "_tmax="+str(tmax)

outp = jVMC.util.OutputManager("../data/output_"+param_name+".hdf5", append=True)

# Set up variational wave function
print("initializing network")
# net = ResNet(
#         F=(filter_size,),
#         channels=(num_hidden,),
#         strides=(1,),
#         bias=False, 
# )

sample_shape = (L,L)

net = CpxRBMCNNLog(
        F=(filter_size,filter_size),
        channels=(num_hidden,),
        strides=(1,1),
        bias=False, 
        periodicBoundary=False,
)

psi = jVMC.vqs.NQS( 
        net, 
        logarithmic=True, 
        seed=4321,
)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for x in range(L):
    for y in range(L):
        hamiltonian.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id((x+1)%L,y,L)))))
        hamiltonian.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id(x,(y+1)%L,L)))))

        hamiltonian.add(op.scal_opstr(g, (op.Sx(xy_to_id(x,y,L)), )))
        hamiltonian.add(op.scal_opstr(h, (op.Sz(xy_to_id(x,y,L)), )))

# Set up observables
observables = {
    "energy": hamiltonian,
    "Z": jVMC.operator.BranchFreeOperator(),
    "ZZ1": jVMC.operator.BranchFreeOperator(),
    "ZZ2": jVMC.operator.BranchFreeOperator(),
    "X": jVMC.operator.BranchFreeOperator(),
    "XX1": jVMC.operator.BranchFreeOperator(),
    "XX2": jVMC.operator.BranchFreeOperator(),
}
for x in range(L):
    for y in range(L):
        observables["X"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), )))
        observables["Z"].add(op.scal_opstr(1. / (L*L), (op.Sz(xy_to_id(x,y,L)), )))

        observables["ZZ1"].add(op.scal_opstr(1. / (L*L), (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id((x+1)%L,y,L)))))
        observables["ZZ1"].add(op.scal_opstr(1. / (L*L), (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id(x,(y+1)%L,L)))))

        observables["XX1"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), op.Sx(xy_to_id((x+1)%L,y,L)))))
        observables["XX1"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), op.Sx(xy_to_id(x,(y+1)%L,L)))))

        observables["ZZ2"].add(op.scal_opstr(1. / (L*L), (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id((x+2)%L,y,L)))))
        observables["ZZ2"].add(op.scal_opstr(1. / (L*L), (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id(x,(y+2)%L,L)))))

        observables["XX2"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), op.Sx(xy_to_id((x+2)%L,y,L)))))
        observables["XX2"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), op.Sx(xy_to_id(x,(y+2)%L,L)))))


# Set up sampler
exactSampler = jVMC.sampler.ExactSampler(psi, sample_shape)
psi2ObsSampler = jVMC.sampler.MCSampler(psi, sample_shape, random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L*L,
                                 numSamples=20000, thermalizationSweeps=25)
# psi2sampler = jVMC.sampler.MCSampler(psi, (L*L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
#                                  numChains=25, sweepSteps=L*L,
#                                  numSamples=numSamples, thermalizationSweeps=25)
# uniformSampler = UniformSampler(psi, (L*L,), numSamples=numSamples)

params = psi.get_parameters()
print("Number of parameters: ", params.size)

######### GS Search ################

# Set u GS hamiltonian
H_GS = jVMC.operator.BranchFreeOperator()
for x in range(L):
    for y in range(L):
        H_GS.add(op.scal_opstr(-1.0, (op.Sx(xy_to_id(x,y,L)), )))


# Set up TDVP
tdvpEquation = jVMC.util.TDVP(exactSampler, rhsPrefactor=1., pinvTol=1e-8, diagonalShift=10, makeReal='real')
# tdvpEquation = tdvp_imp.TDVP({"lhs": exactSampler, "rhs": exactSampler}, rhsPrefactor=1., pinvTol=1e-8, diagonalShift=10, makeReal='real')

print("starting GS search")
# jVMC.util.ground_state_search(psi, H_GS, tdvpEquation, exactSampler, numSteps=50)

#####################################

print("setting up tdvp equation")
tdvpEquation = jVMC.util.TDVP(exactSampler, rhsPrefactor=1.j)

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)

t = 0.
# Measure initial observables
parameters = []
params = psi.get_parameters()
parameters.append(params) 
# obs = measure(observables, psi, exactSampler)
data = []
# data.append([t, 
#     obs["energy"]["mean"][0], 
#     obs["energy"]["variance"][0], 
#     obs["energy"]["MC_error"][0], 
#     obs["Z"]["mean"][0],
#     obs["Z"]["variance"][0], 
#     obs["Z"]["MC_error"][0], 
#     obs["ZZ1"]["mean"][0],
#     obs["ZZ1"]["variance"][0], 
#     obs["ZZ1"]["MC_error"][0], 
#     obs["ZZ2"]["mean"][0],
#     obs["ZZ2"]["variance"][0], 
#     obs["ZZ2"]["MC_error"][0], 
#     obs["X"]["mean"][0],
#     obs["X"]["variance"][0], 
#     obs["X"]["MC_error"][0], 
#     obs["XX1"]["mean"][0],
#     obs["XX1"]["variance"][0], 
#     obs["XX1"]["MC_error"][0], 
#     obs["XX2"]["mean"][0],
#     obs["XX2"]["variance"][0], 
#     obs["XX2"]["MC_error"][0], 
#     0, 0, 0])


print("starting tdvp equation")
while t < tmax:
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
    print("    xPol = %f +/- %f" % (obs["X"]["mean"][0], obs["X"]["MC_error"][0]))
    toc = time.perf_counter()
    print("   == Total time for this step: %fs\n" % (toc - tic))

    data.append([t, 
        obs["energy"]["mean"][0], 
        obs["energy"]["variance"][0], 
        obs["energy"]["MC_error"][0], 
        obs["Z"]["mean"][0],
        obs["Z"]["variance"][0], 
        obs["Z"]["MC_error"][0], 
        obs["ZZ1"]["mean"][0],
        obs["ZZ1"]["variance"][0], 
        obs["ZZ1"]["MC_error"][0], 
        obs["ZZ2"]["mean"][0],
        obs["ZZ2"]["variance"][0], 
        obs["ZZ2"]["MC_error"][0], 
        obs["X"]["mean"][0],
        obs["X"]["variance"][0], 
        obs["X"]["MC_error"][0], 
        obs["XX1"]["mean"][0],
        obs["XX1"]["variance"][0], 
        obs["XX1"]["MC_error"][0], 
        obs["XX2"]["mean"][0],
        obs["XX2"]["variance"][0], 
        obs["XX2"]["MC_error"][0], 
        tdvpErr, tdvpRes, dt])

    params = psi.get_parameters()
    parameters.append(params) 

    npdata   = np.array(data)
    npparams = np.array(parameters)

    dfTDVP = pd.DataFrame( {
        "time":       npdata[:, 0],
        "energy":     npdata[:, 1],
        "energy_var": npdata[:, 2],
        "energy_MC":  npdata[:, 3],
        "zPol":       npdata[:, 4],
        "zPol_var":   npdata[:, 5],
        "zPol_MC":    npdata[:, 6],
        "zz1":        npdata[:, 7],
        "zz1_var":    npdata[:, 8],
        "zz1_MC":     npdata[:, 9],
        "zz2":        npdata[:, 10],
        "zz2_var":    npdata[:, 11],
        "zz2_MC":     npdata[:, 12],
        "xPol":       npdata[:, 13],
        "xPol_var":   npdata[:, 14],
        "xPol_MC":    npdata[:, 15],
        "xx1":        npdata[:, 16],
        "xx1_var":    npdata[:, 17],
        "xx1_MC":     npdata[:, 18],
        "xx2":        npdata[:, 19],
        "xx2_var":    npdata[:, 20],
        "xx2_MC":     npdata[:, 21],
        "tdvpErr":    npdata[:, 22],
        "tdvpRes":    npdata[:, 23],
        "dt":         npdata[:, 24],
    })

    # dfTDVP.to_csv("../data/data_"+param_name+".csv", sep=' ')

    # with h5py.File("../data/data_"+param_name+".h5", 'w') as f:
    #     # If single array, save directly
    #     f.create_dataset("time",       data=npdata[:,0])
    #     f.create_dataset("energy",     data=npdata[:,1])
    #     f.create_dataset("energy_var", data=npdata[:,2])
    #     f.create_dataset("energy_MC",  data=npdata[:,3])
    #     f.create_dataset("zPol",       data=npdata[:,4])
    #     f.create_dataset("zPol_var",   data=npdata[:,5])
    #     f.create_dataset("zPol_MC",    data=npdata[:,6])
    #     f.create_dataset("zz1",        data=npdata[:,7])
    #     f.create_dataset("zz1_var",    data=npdata[:,8])
    #     f.create_dataset("zz1_MC",     data=npdata[:,9])
    #     f.create_dataset("zz2",        data=npdata[:,10])
    #     f.create_dataset("zz2_var",    data=npdata[:,11])
    #     f.create_dataset("zz2_MC",     data=npdata[:,12])
    #     f.create_dataset("xPol",       data=npdata[:,13])
    #     f.create_dataset("xPol_var",   data=npdata[:,14])
    #     f.create_dataset("xPol_MC",    data=npdata[:,15])
    #     f.create_dataset("xx1",        data=npdata[:,16])
    #     f.create_dataset("xx1_var",    data=npdata[:,17])
    #         # If list of arrays, create a group and save each array
    #     grp = f.create_group("params")
    #     for i, arr in enumerate(npparams):
    #         grp.create_dataset(f'params_{i}', data=arr)


tic = time.perf_counter()
print(">  t = %f\n" % (t))
print("done")

data = np.array(data)
fig, axs = plt.subplots(2)
#plt.ylim(data[-1,2], 1)

df = pd.read_csv('ref_L=10.csv')
axs[1].plot(df['time'], df['xPol'], color='red')
axs[1].plot(data[:,0], data[:,2])
axs[0].plot(data[:,0], data[:,1])
axs[1].set_xlim(0,tmax+0.1)
plt.savefig("plot.pdf")

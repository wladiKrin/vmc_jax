import os
print(os.environ['CONDA_DEFAULT_ENV'])

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
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

import new_tdvp
from RWKV import CpxRWKV

from functools import partial

import json
import sys

def xy_to_id(x,y,L):
    return int(x + L * y)

class UniformSampler:

    def __init__(self, net, sampleShape, key=123, numSamples=100):
        self.sampleShape = sampleShape
        self.net = net
        if isinstance(key,jax.Array):
            self.key = key
        else:
            self.key = jax.random.PRNGKey(key)
        # self.key = jax.random.split(self.key, mpi.commSize)[mpi.rank]
        # self.key = jax.random.split(self.key, global_defs.device_count())
    
        self.numSamples = numSamples
        self.lastNumSamples = 0

        # Make sure that net is initialized
        self.net(jnp.zeros((jVMC.global_defs.device_count(), 1) + sampleShape, dtype=np.int32))

    def sample(self, parameters=None, numSamples=None, multipleOf=1):
        if numSamples is None:
            numSamples = self.numSamples
        numSamples, self.lastNumSamples = mpi.distribute_sampling(numSamples, localDevices=jVMC.global_defs.device_count(), numChainsPerDevice=1)
        # numSamples = mpi.distribute_sampling(numSamples, localDevices=jVMC.global_defs.device_count(), numChainsPerDevice=1)
        key, self.key = jax.random.split(self.key)
        configs = 1 * jax.random.bernoulli(key, shape=(1,numSamples,)+self.sampleShape)
        coeffs = self.net(configs)
        return configs, coeffs, jnp.ones(configs.shape[:2]) * (2.0**(np.prod(self.sampleShape)) / self.lastNumSamples)
    
    def get_last_number_of_samples(self):
        return self.lastNumSamples

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

dt = 1e-4  # Initial time step
integratorTol = inp["integratorTol"]  # Adaptive integrator tolerance
tmax = inp["tmax"]  # Final time
numSamples = inp["numSamples"]
num_layers=inp["num_layers"]
embedding_size=inp["embedding_size"]
hidden_size=inp["hidden_size"]
num_heads=inp["num_heads"]

outp = jVMC.util.OutputManager(inp["data_dir"]+"/data2DMixed_L="+str(L)+"_g="+str(g)+"_num_layers="+str(num_layers)+"_embedding_size="+str(embedding_size)+"_hidden_size="+str(hidden_size)+"_num_heads="+str(num_heads)+"_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+"_tmax="+str(tmax)+".hdf5", append=True)

# Set up variational wave function
#net = jVMC.nets.RNN1DGeneral(L=L, hiddenSize=hiddenSize, depth=depth)
#net = CpxRWKV(L=L, embedding_size=hiddenSize, numLayers=depth+1)
net = CpxRWKV(L=L*L, num_layers=num_layers, embedding_size=embedding_size, hidden_size=hidden_size, num_heads=num_heads, bias=True, one_hot=True, patch_size=4)

psi = jVMC.vqs.NQS(net, logarithmic=True, seed=1234)  # Variational wave function
# psi = jVMC.vqs.NQS(net, seed=1234)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for x in range(L):
    for y in range(L):
        if x+1 < L:
            hamiltonian.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id(x+1,y,L)))))
        if y+1 < L:
            hamiltonian.add(op.scal_opstr(-1., (op.Sz(xy_to_id(x,y,L)), op.Sz(xy_to_id(x,y+1,L)))))
        hamiltonian.add(op.scal_opstr(g, (op.Sx(xy_to_id(x,y,L)), )))
        hamiltonian.add(op.scal_opstr(h, (op.Sz(xy_to_id(x,y,L)),)))

print(hamiltonian)

# Set u GS hamiltonian
H_GS = jVMC.operator.BranchFreeOperator()
for x in range(L):
    for y in range(L):
        H_GS.add(op.scal_opstr(-1.0, (op.Sx(xy_to_id(x,y,L)), )))

# Set up observables
observables = {
    "energy": hamiltonian,
    "X": jVMC.operator.BranchFreeOperator(),
}
for x in range(L):
    for y in range(L):
        observables["X"].add(op.scal_opstr(1. / (L*L), (op.Sx(xy_to_id(x,y,L)), )))

# Set up sampler
# exactSampler = jVMC.sampler.ExactSampler(psi, L)
psi2sampler = jVMC.sampler.MCSampler(psi, (L*L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L*L,
                                 numSamples=numSamples, thermalizationSweeps=25)
uniformSampler = UniformSampler(psi, (L*L,), numSamples=numSamples)

params = psi.get_parameters()
print("Number of parameters: ", params.size)

######### GS Search #################
# t, weights = outp.get_network_checkpoint(0)
# psi.set_parameters(weights)
# Set up sampler
sampler = jVMC.sampler.MCSampler(psi, (L*L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L*L,
                                 numSamples=2000, thermalizationSweeps=25)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   pinvTol=1e-8, diagonalShift=10, makeReal='real')

print("starting GS search")
jVMC.util.ground_state_search(psi, H_GS, tdvpEquation, sampler, numSteps=200)

#####################################

print("network checkpoint")
outp.write_network_checkpoint(0.0, psi.get_parameters())

# Get sample
# numSamples=1000
# sampleConfigs, sampleLogPsi, p = exactSampler.sample()

# Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, sampleLogPsi, t)
# Eloc = SampledObs( Eloc, p)
# sampleGradients = psi.gradients(sampleConfigs)
# sampleGradients = SampledObs( sampleGradients, p)
# # sampleGradients = sampleGradients * jnp.exp(sampleLogPsi[:,:,None])

# S = 1.j * jnp.imag( sampleGradients.covar() )
# # S = mpi.global_sum(jVMC.stats._covar_helper(sampleGradients, sampleGradients)[:,None,...])
# S = 1.j * jnp.imag( S )

# ev, V = jnp.linalg.eigh(S)

# # print(jnp.imag(-1.j*sampleGradients.covar(Eloc)))

# sampleConfigs, sampleLogPsi, p = uniformSampler.sample(numSamples=numSamples)
# sampleGradients = psi.gradients(sampleConfigs)
# sampleGradients = sampleGradients * jnp.exp(sampleLogPsi[:,:,None])

# S = mpi.global_sum(jVMC.stats._covar_helper(sampleGradients, sampleGradients)[:,None,...]) * p.ravel()[0]

# S = 1.j * jnp.imag( S )

# ev, V = jnp.linalg.eigh(S)

# sampleConfigs, sampleLogPsi, p = psi2sampler.sample(numSamples=numSamples)
# sampleGradients = psi.gradients(sampleConfigs)
# sampleGradients = SampledObs(sampleGradients, p)
# Eloc = hamiltonian.get_O_loc(sampleConfigs, psi, sampleLogPsi, t)
# Eloc = SampledObs( Eloc, p)

# F = sampleGradients.covar(Eloc)
# # print(jnp.imag(-1.j*F))
# print(sampleGradients.covar())

# exit()

# print(ev)
# exit()

# Set up TDVP
#tdvpEquation = new_tdvp.TDVP({"lhs": uniformSampler, "rhs":psi2sampler}, pinvTol=inp["pinvTol"], pinvCutoff=1e-4,
#                                   rhsPrefactor=1.j,
#                                   makeReal='real')
# tdvpEquation = new_tdvp.TDVP({"lhs": uniformSampler, "rhs":psi2sampler}, **inp["tdvp"], rhsPrefactor=1.j)
print("setting up tdvp equation")
tdvpEquation = new_tdvp.TDVP({"lhs": uniformSampler, "rhs":psi2sampler}, rhsPrefactor=1.j)

t = 0.0  # Initial time

# Set up stepper
stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol)

# Measure initial observables
obs = measure(observables, psi, psi2sampler)
data = []
data.append([t, obs["energy"]["mean"][0], obs["X"]["mean"][0]])

def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))

print("starting tdvp equation")
while t < tmax:
    tic = time.perf_counter()
    print(">  t = %f\n" % (t))

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, 
                          outp=outp, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    psi.set_parameters(dp)
    t += dt

    # Measure observables
    obs = measure(observables, psi, psi2sampler)
    data.append([t, obs["energy"]["mean"][0], obs["X"]["mean"][0]])

    # Write some meta info to screen
    outp.print("   Time step size: dt = %f" % (dt))
    tdvpErr, tdvpRes = tdvpEquation.get_residuals()
    outp.print("   Residuals: tdvp_err = %.2e, solver_res = %.2e" % (tdvpErr, tdvpRes))
    outp.print("    Energy = %f +/- %f" % (obs["energy"]["mean"][0], obs["energy"]["MC_error"][0]))
    toc = time.perf_counter()
    outp.print("   == Total time for this step: %fs\n" % (toc - tic))

    npdata = np.array(data)
    dfTDVP = pd.DataFrame(
        {
            "time": npdata[:, 0],
            "energy": npdata[:, 1],
            "xPol": npdata[:, 2],
            #"xxCorr": npdata[:, 3],
            #"xx2Corr": npdata[:, 4],
            #"xx3Corr": npdata[:, 5],
            #"xx4Corr": npdata[:, 6],
            #"zPol": npdata[:, 5],
            #"zzCorr": npdata[:, 6],
            #"calcTime": npdata[:, 7],
        }
    )
    dfTDVP.to_csv(inp["data_dir"]+"/vmc_tdvp2DMixed_data_L="+str(L)+"_g="+str(g)+"_num_layers="+str(num_layers)+"_embedding_size="+str(embedding_size)+"_hidden_size="+str(hidden_size)+"_num_heads="+str(num_heads)+"_numSamples="+str(numSamples)+"_integratorTol="+str(integratorTol)+"_tmax="+str(tmax)+".csv", sep=' ')


tic = time.perf_counter()
outp.print(">  t = %f\n" % (t))
outp.print("done")

data = np.array(data)
plt.plot(data[:,0], data[:,2])
plt.savefig("plot.pdf")

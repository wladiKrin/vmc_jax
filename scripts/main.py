import os

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
import jax.numpy as jnp

import numpy as np

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
g = -1.0
h = 0.0

dt = 1e-3  # Initial time step
integratorTol = 1e-4  # Adaptive integrator tolerance
tmax = 1.0  # Final time
numSamples = inp["numSamples"]

hiddenSize=inp["hiddenSize"]
depth=inp["depth"]

outp = jVMC.util.OutputManager(inp["data_dir"]+"/data_L="+str(L)+"_hiddenSize="+str(hiddenSize)+"_depth="+str(depth)+".hdf5", append=True)

# Set up variational wave function
#net = jVMC.nets.RNN1DGeneral(L=L, hiddenSize=hiddenSize, depth=depth)
net = CpxRWKV(L=L, embedding_size=hiddenSize, num_layers=depth+1)

psi = jVMC.vqs.NQS(net, logarithmic=True, seed=1234)  # Variational wave function
# psi = jVMC.vqs.NQS(net, seed=1234)  # Variational wave function

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(op.scal_opstr(-1., (op.Sz(l), op.Sz((l + 1) % L))))
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
psi2sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=25, sweepSteps=L,
                                 numSamples=numSamples, thermalizationSweeps=25)
uniformSampler = UniformSampler(psi, (L,), numSamples=numSamples)

params = psi.get_parameters()
print("Number of parameters: ", params.size)

######### GS Search #################
# t, weights = outp.get_network_checkpoint(0)
# psi.set_parameters(weights)
# Set up sampler
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=2000, thermalizationSweeps=25)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   pinvTol=1e-8, diagonalShift=10, makeReal='real')

jVMC.util.ground_state_search(psi, H_GS, tdvpEquation, sampler, numSteps=200)

#####################################

outp.write_network_checkpoint(0.0, psi.get_parameters())

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

tic = time.perf_counter()
outp.print(">  t = %f\n" % (t))
outp.print("done")

data = np.array(data)
plt.plot(data[:,0], data[:,2])
plt.savefig("plot.pdf")

import jax
import jax.random as random
import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.mpi_wrapper as mpi
import jVMC.global_defs as global_defs

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

        exactSampler = jVMC.sampler.ExactSampler(self.net, np.prod(self.sampleShape))
        renormEx = 1/exactSampler.get_norm()

        # configs = exactSampler.basis
        coeffs = self.net(configs)
        
        weights = jnp.ones(configs.shape[:2]) * (2.0**(-np.prod(self.sampleShape)))
        
        # renorm1 = self.lastNumSamples / jnp.sum(jnp.abs(jnp.exp(coeffs))**2 / weights)
        # renorm2 = 1/exactSampler.get_norm()
        # renorm = renorm2


        # print("renorm: ", renorm)

        if self.net.logarithmic:
            renorm = self.lastNumSamples / jnp.sum(jnp.abs(jnp.exp(coeffs))**2 / weights)
            # renormEx = 1/exactSampler.get_norm()
            return configs, coeffs, renorm * jnp.abs(jnp.exp(coeffs))**2 / self.lastNumSamples / weights
        else:
            # renorm = renormEx #self.lastNumSamples / jnp.sum(jnp.abs(coeffs)**2 / weights)
            renorm = self.lastNumSamples / jnp.sum(jnp.abs(coeffs)**2 / weights)
            # print("r: ", renorm)
            # print("rEx: ", renormEx)
            return configs, coeffs, renorm / self.lastNumSamples / weights
    
    def get_last_number_of_samples(self):
        return self.lastNumSamples

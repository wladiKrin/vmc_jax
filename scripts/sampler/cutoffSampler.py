import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap

import jVMC
import jVMC.mpi_wrapper as mpi
from jVMC.nets.sym_wrapper import SymNet

from functools import partial

import jVMC.global_defs as global_defs


def propose_spin_flip(key, s, info):
    idx = random.randint(key, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1) % 2
    return s.at[idx].set(update)


def propose_POVM_outcome(key, s, info):
    key, subkey = random.split(key)
    idx = random.randint(subkey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + random.randint(key, (1,), 0, 3) % 4)
    return s.at[idx].set(update)


def propose_spin_flip_Z2(key, s, info):
    idxKey, flipKey = jax.random.split(key)
    idx = random.randint(idxKey, (1,), 0, s.size)[0]
    idx = jnp.unravel_index(idx, s.shape)
    update = (s[idx] + 1) % 2
    s = s.at[idx].set(update)
    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)


def propose_spin_flip_zeroMag(key, s, info):
    # propose spin flips that stay in the zero magnetization sector

    idxKeyUp, idxKeyDown, flipKey = jax.random.split(key, num=3)

    # can't use jnp.where because then it is not jit-compilable
    # find indices based on cumsum
    bound_up = jax.random.randint(idxKeyUp, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)
    bound_down = jax.random.randint(idxKeyDown, (1,), 1, s.shape[0] * s.shape[1] / 2 + 1)

    id_up = jnp.searchsorted(jnp.cumsum(s), bound_up)
    id_down = jnp.searchsorted(jnp.cumsum(1 - s), bound_down)

    idx_up = jnp.unravel_index(id_up, s.shape)
    idx_down = jnp.unravel_index(id_down, s.shape)

    s = s.at[idx_up[0], idx_up[1]].set(0)
    s = s.at[idx_down[0], idx_down[1]].set(1)

    # On average, do a global spin flip every 30 updates to
    # reflect Z_2 symmetry
    doFlip = random.randint(flipKey, (1,), 0, 5)[0]
    return jax.lax.cond(doFlip == 0, lambda x: 1 - x, lambda x: x, s)


class CutoffSampler:
    """A sampler class.

    This class provides functionality to sample computational basis states from \
    the distribution 

        :math:`p_{\\eps}(s)=\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}}`

    for :math:`\\frac{|\\psi(s)|^{\\mu}}{\\sum_s|\\psi(s)|^{\\mu}} > \\eps`

        :math:`\\eps`.

    else.

    Sampling is automatically distributed accross MPI processes and locally available \
    devices.

    Initializer arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis configurations.
        * ``key``: An instance of ``jax.random.PRNGKey``. Alternatively, an ``int`` that will be used \
                   as seed to initialize a ``PRNGKey``.
        * ``eps``: epsilon used for the cutoff procedure.
        * ``updateProposer``: A function to propose updates for the MCMC algorithm. \
        It is called as ``updateProposer(key, config, **kwargs)``, where ``key`` is an instance of \
        ``jax.random.PRNGKey``, ``config`` is a computational basis configuration, and ``**kwargs`` \
        are optional additional arguments.
        * ``numChains``: Number of Markov chains, which are run in parallel.
        * ``updateProposerArg``: An optional argument that will be passed to the ``updateProposer`` \
        as ``kwargs``.
        * ``numSamples``: Default number of samples to be returned by the ``sample()`` member function.
        * ``thermalizationSweeps``: Number of sweeps to perform for thermalization of the Markov chain.
        * ``sweepSteps``: Number of proposed updates per sweep.
        * ``mu``: Parameter for the distribution :math:`p_{\\mu}(s)`, see above.
        * ``logProbFactor``: Factor for the log-probabilities, aquivalent to the exponent for the probability \
        distribution. For pure wave functions this should be 0.5, and 1.0 for POVMs. In the POVM case, the \
        ``mu`` parameter must be set to 1.0, to sample the unchanged POVM distribution.
    """

    def __init__(self, net, sampleShape, key, eps, updateProposer=None, numChains=1, updateProposerArg=None,
                 numSamples=100, thermalizationSweeps=10, sweepSteps=10, initState=None, mu=2, logProbFactor=0.5):
        """Initializes the MCSampler class.

        """

        self.sampleShape = sampleShape

        self.net = net
        self.eps = eps
        if (not net.is_generator) and (updateProposer is None):
            raise RuntimeError("Instantiation of MCSampler: `updateProposer` is `None` and cannot be used for MCMC sampling.")
        self.orbit = None
        if isinstance(self.net.net, SymNet):
            self.orbit = self.net.net.orbit.orbit

        stateShape = (global_defs.device_count(), numChains) + sampleShape
        if initState is None:
            initState = jnp.zeros(sampleShape, dtype=np.int32)
        self.states = jnp.stack([initState] * (global_defs.device_count() * numChains), axis=0).reshape(stateShape)

        # Make sure that net is initialized
        self.net(self.states)

        self.logProbFactor = logProbFactor
        self.mu = mu
        if mu < 0 or mu > 2:
            raise ValueError("mu must be in the range [0, 2]")
        self.updateProposer = updateProposer
        self.updateProposerArg = updateProposerArg

        if isinstance(key,jax.Array):
            self.key = key
        else:
            self.key = jax.random.PRNGKey(key)
        self.key = jax.random.split(self.key, mpi.commSize)[mpi.rank]
        self.key = jax.random.split(self.key, global_defs.device_count())
        self.thermalizationSweeps = thermalizationSweeps
        self.sweepSteps = sweepSteps
        self.numSamples = numSamples

        shape = (global_defs.device_count(),) + (1,)
        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

        self.numChains = numChains

        ## based on exact norm
        # self.normSampler = jVMC.sampler.ExactSampler(self.net, np.prod(self.sampleShape))
        # self.cutoff = jnp.log(self.normSampler.get_norm() * self.eps)

        ## based on sampled max coeff
        sampler = jVMC.sampler.MCSampler(self.net, (np.prod(self.sampleShape),), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                     numChains=25, sweepSteps=np.prod(self.sampleShape),
                                     numSamples=numSamples, thermalizationSweeps=50)
        _,coeffs,_ = sampler.sample()
        self.cutoff = jnp.log(jnp.max(jnp.abs(coeffs)**2)* self.eps)

        # jit'd member functions
        self._get_samples_jitd = {}  # will hold a jit'd function for each number of samples
        self._randomize_samples_jitd = {}  # will hold a jit'd function for each number of samples

    def set_number_of_samples(self, N):
        """Set default number of samples.

        Arguments:
            * ``N``: Number of samples.
        """

        self.numSamples = N

    def set_random_key(self, key):
        """Set key for pseudo random number generator.

        Args:
            * ``key``: Key (jax.random.PRNGKey)
        """

        self.key = jax.random.split(key, global_defs.device_count())

    def get_last_number_of_samples(self):
        """Return last number of samples.

        This function is required, because the actual number of samples might \
        exceed the requested number of samples when sampling is distributed \
        accross multiple processes or devices.

        Returns:
            Number of samples generated by last call to ``sample()`` member function.
        """
        return self.globNumSamples

    def sample(self, parameters=None, numSamples=None, multipleOf=1):
        """Generate random samples from wave function.

        If supported by ``net``, direct sampling is peformed. Otherwise, MCMC is run \
        to generate the desired number of samples. For direct sampling the real part \
        of ``net`` needs to provide a ``sample()`` member function that generates \
        samples from :math:`p_{\\mu}(s)`.

        Sampling is automatically distributed accross MPI processes and available \
        devices. In that case the number of samples returned might exceed ``numSamples``.

        Arguments:
            * ``parameters``: Network parameters to use for sampling.
            * ``numSamples``: Number of samples to generate. When running multiple processes \
            or on multiple devices per process, the number of samples returned is \
            ``numSamples`` or more. If ``None``, the default number of samples is returned \
            (see ``set_number_of_samples()`` member function).
            * ``multipleOf``: This argument allows to choose the number of samples returned to \
            be the smallest multiple of ``multipleOf`` larger than ``numSamples``. This feature \
            is useful to distribute a total number of samples across multiple processors in such \
            a way that the number of samples per processor is identical for each processor.

        Returns:
            A sample of computational basis configurations drawn from :math:`p_{\\mu}(s)`.
        """

        if numSamples is None:
            numSamples = self.numSamples

        if self.net.is_generator:
            if parameters is not None:
                tmpP = self.net.params
                self.net.set_parameters(parameters)
            configs, coeffs, ps = self._get_samples_gen(self.net.parameters, numSamples, multipleOf)
            if parameters is not None:
                self.net.params = tmpP
            return configs, coeffs, ps

        configs, logPsi = self._get_samples_mcmc(parameters, numSamples, multipleOf)

        ## based on exact norm
        # self.cutoff = jnp.log(self.normSampler.get_norm() * self.eps)

        ## based on sampled max coeff
        self.cutoff = jnp.log(jnp.max(jnp.abs(logPsi)**2) * self.eps)

        p = jnp.exp((1.0 / self.logProbFactor - self.mu) * jnp.real(jnp.log(logPsi)))

        weights = jnp.where(jnp.abs(logPsi)**2 > jnp.exp(self.cutoff), jnp.abs(logPsi)**2, jnp.exp(self.cutoff))
        ratio = jnp.abs(logPsi)**2 / weights

        renorm = numSamples / jnp.sum(ratio)
        print("r: ", renorm)
        # print("rEx: ", renormEx)
        return configs, logPsi, (p/mpi.global_sum(p)) * renorm * ratio #/ self.lastNumSamples / weights

        # return configs, logPsi, p / mpi.global_sum(p)

    def _randomize_samples(self, samples, key, orbit):
        """ For a given set of samples apply a random symmetry transformation to each sample
        """
        orbit_indices = random.choice(key, orbit.shape[0], shape=(samples.shape[0],))
        samples = samples * 2 - 1
        return jax.vmap(lambda o, idx, s: (o[idx].dot(s.ravel()).reshape(s.shape) + 1) // 2, in_axes=(None, 0, 0))(orbit, orbit_indices, samples)

    def _get_samples_gen(self, params, numSamples, multipleOf=1):

        numSamples, self.globNumSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=multipleOf)

        tmpKeys = random.split(self.key[0], 3 * global_defs.device_count())
        self.key = tmpKeys[:global_defs.device_count()]
        tmpKey = tmpKeys[global_defs.device_count():2 * global_defs.device_count()]
        tmpKey2 = tmpKeys[2 * global_defs.device_count():]

        samples = self.net.sample(numSamples, tmpKey, parameters=params)

        if not str(numSamples) in self._randomize_samples_jitd:
            self._randomize_samples_jitd[str(numSamples)] = global_defs.pmap_for_my_devices(self._randomize_samples, static_broadcasted_argnums=(), in_axes=(0, 0, None))

        if not self.orbit is None:
            samples = self._randomize_samples_jitd[str(numSamples)](samples, tmpKey2, self.orbit)
            # return self._randomize_samples_jitd[str(numSamples)](samples, tmpKey2, self.orbit)
        
        return samples, self.net(samples), jnp.ones(samples.shape[:2]) / self.globNumSamples

    def _get_samples_mcmc(self, params, numSamples, multipleOf=1):

        tmpP = None
        if params is not None:
            tmpP = self.net.params
            self.net.set_parameters(params)

        net, params = self.net.get_sampler_net()

        if tmpP is not None:
            self.net.params = tmpP        

        # Initialize sampling stuff
        self._mc_init(params)

        numSamples, self.globNumSamples = mpi.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=np.lcm(self.numChains, multipleOf))
        numSamplesStr = str(numSamples)

        # check whether _get_samples is already compiled for given number of samples
        if not numSamplesStr in self._get_samples_jitd:
            self._get_samples_jitd[numSamplesStr] = global_defs.pmap_for_my_devices(partial(self._get_samples, sweepFunction=partial(self._sweep, net=net)),
                                                                                    static_broadcasted_argnums=(1, 2, 3, 9, 11),
                                                                                    in_axes=(None, None, None, None, 0, 0, 0, 0, 0, None, None, None))

        (self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted), configs =\
            self._get_samples_jitd[numSamplesStr](params, numSamples, self.thermalizationSweeps, self.sweepSteps,
                                                  self.states, self.logAccProb, self.key, self.numProposed, self.numAccepted,
                                                  self.updateProposer, self.updateProposerArg, self.sampleShape)

        tmpP = self.net.params
        self.net.params = params["params"]
        coeffs = self.net(configs)
        self.net.params = tmpP

        # return configs, None
        return configs, coeffs

    def _get_samples(self, params, numSamples,
                     thermSweeps, sweepSteps,
                     states, logAccProb, key,
                     numProposed, numAccepted,
                     updateProposer, updateProposerArg,
                     sampleShape, sweepFunction=None):

        # Thermalize
        states, logAccProb, key, numProposed, numAccepted =\
            sweepFunction(states, logAccProb, key, numProposed, numAccepted, params, thermSweeps * sweepSteps, updateProposer, updateProposerArg)

        # Collect samples
        def scan_fun(c, x):

            states, logAccProb, key, numProposed, numAccepted =\
                sweepFunction(c[0], c[1], c[2], c[3], c[4], params, sweepSteps, updateProposer, updateProposerArg)

            return (states, logAccProb, key, numProposed, numAccepted), states

        meta, configs = jax.lax.scan(scan_fun, (states, logAccProb, key, numProposed, numAccepted), None, length=numSamples)

        # return meta, configs.reshape((configs.shape[0]*configs.shape[1], -1))
        return meta, configs.reshape((configs.shape[0] * configs.shape[1],) + sampleShape)

    def _sweep(self, states, logAccProb, key, numProposed, numAccepted, params, numSteps, updateProposer, updateProposerArg, net=None):

        def perform_mc_update(i, carry):

            # Generate update proposals
            newKeys = random.split(carry[2], carry[0].shape[0] + 1)
            carryKey = newKeys[-1]
            newStates = vmap(updateProposer, in_axes=(0, 0, None))(newKeys[:len(carry[0])], carry[0], updateProposerArg)

            # Compute acceptance probabilities
            # if self.net.logarithmic:
            #     newLogAccProb = jax.vmap(lambda y: self.mu * jnp.real(net(params, y)), in_axes=(0,))(newStates)
            #     P = jnp.exp(newLogAccProb - carry[1])
            # else:
            #     newLogAccProb = jax.vmap(lambda y: net(params, y)**self.mu, in_axes=(0,))(newStates)
            #     P = newLogAccProb / carry[1]

            newLogAccProb = jax.vmap(lambda y: self.mu * jnp.real(net(params, y)), in_axes=(0,))(newStates)

            # def cutoff(acc, prob)
            #     return jax.lax.cond(acc, lambda x: x, lambda x: self.normalization * eps, prob)
            newLogAccProb = jnp.where(newLogAccProb > self.cutoff, newLogAccProb, self.cutoff)
            P = jnp.exp(newLogAccProb - carry[1])

            # Roll dice
            newKey, carryKey = random.split(carryKey,)
            accepted = random.bernoulli(newKey, P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[3] + len(newStates)
            numAccepted = carry[4] + jnp.sum(accepted)

            # Perform accepted updates
            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old, new))
            carryStates = vmap(update, in_axes=(0, 0, 0))(accepted, carry[0], newStates)

            carryLogAccProb = jnp.where(accepted == True, newLogAccProb, carry[1])

            return (carryStates, carryLogAccProb, carryKey, numProposed, numAccepted)

        (states, logAccProb, key, numProposed, numAccepted) =\
            jax.lax.fori_loop(0, numSteps, perform_mc_update, (states, logAccProb, key, numProposed, numAccepted))

        return states, logAccProb, key, numProposed, numAccepted

    def _mc_init(self, netParams):

        # Initialize logAccProb
        net, _ = self.net.get_sampler_net()
        # if self.net.logarithmic:
        #     self.logAccProb = global_defs.pmap_for_my_devices(
        #         lambda x: jax.vmap(lambda y: self.mu * jnp.real(net(netParams, y)), in_axes=(0,))(x)
        #     )(self.states)
        # else:
        #     self.logAccProb = global_defs.pmap_for_my_devices(
        #         lambda x: jax.vmap(lambda y:  jnp.abs(net(netParams, y))**self.mu, in_axes=(0,))(x)
        #
        #     )(self.states)

        self.logAccProb = global_defs.pmap_for_my_devices(
            lambda x: jax.vmap(lambda y: self.mu * jnp.real(net(netParams, y)), in_axes=(0,))(x)
        )(self.states)

        self.logAccProb = jnp.where(self.logAccProb > self.cutoff, self.logAccProb, self.cutoff)

        shape = (global_defs.device_count(),) + (1,)

        self.numProposed = jnp.zeros(shape, dtype=np.int64)
        self.numAccepted = jnp.zeros(shape, dtype=np.int64)

    def acceptance_ratio(self):
        """Get acceptance ratio.

        Returns:
            Acceptance ratio observed in the last call to ``sample()``.
        """

        numProp = mpi.global_sum(self.numProposed)
        if numProp > 0:
            return mpi.global_sum(self.numAccepted) / numProp

        return jnp.array([0.])

# ** end class Sampler

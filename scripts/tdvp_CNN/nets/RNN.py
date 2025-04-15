from jax import numpy as jnp
from jax import random as jrnd
from jax import jit
import jax

import jVMC
import h5py as h5

from typing import Any, List, Optional, Tuple
from functools import partial
from jax import Array, vmap, debug, jit
from itertools import repeat, product

import flax.linen as nn

class RNN(nn.Module):
    # Main
    L: int = 10
    LocalHilDim: int = 2 # local hilbert space dimension for spinful fermions
    dtype: type = jVMC.util.global_defs.tReal
    # time/channel mixing
    features: int = 4
    # prob correction
    logProbFactor: float = 0.5

    def setup(self):

        # encoding layer
        self.layer0 = nn.Dense(self.features)
        # the lstm takes the input directly
        self.cell = nn.RNN(nn.LSTMCell(features=self.features), return_carry=True) #, cell_size=self.features)
        self.amplitude = nn.Dense(self.LocalHilDim)
        self.phase = nn.Dense(1)
        # quantum number masks

    def __call__(self, s: Array, carry_state: Array = None, output_state: bool = False) -> Array:
        # the helper method allows to use nn.jit with static_argnames
        return self.forward_with_state(s, carry_state, output_state)

    @partial(nn.jit, static_argnums=3)
    def forward_with_state(self, s: Array, carry_state: Array = None,  output_state: bool = False) -> Array:

        # embed the input and apply the input layer norm
        if output_state:
            seqy = jax.nn.one_hot(s,self.LocalHilDim)
            # apply encoding layer
            pos_seqy = self.layer0(seqy)
        else:
            seqy = jax.nn.one_hot(jnp.pad(s,(1,0)),self.LocalHilDim)
            # apply encoding layer
            pos_seqy = jax.vmap(self.layer0)(seqy)

        # return here for RNN mode
        if output_state:
            carry_state, y = self.cell(jnp.expand_dims(pos_seqy,0), initial_carry=carry_state)
            # apply FFN layer
            return self.amplitude(y).ravel(), carry_state
        
        carry_state = (jnp.zeros((1,self.features)),jnp.zeros((1,self.features)),)
        carry_state, y = self.cell(pos_seqy, initial_carry=carry_state)
        x = jax.vmap(self.amplitude)(y[:-1].squeeze(1))
        phase = self.phase(y[-1].ravel())

        #####################################
        x = nn.log_softmax(x) * self.logProbFactor
        # the log-probs according the state
        return (jnp.take_along_axis(x, jnp.expand_dims(
                                s, # here we shift the state by one to match the indexing
                                axis=-1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1) 
                                + 1.j * phase)[0]

    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = jrnd.split(key, self.L)
            logits, carry = self(jnp.zeros(1,dtype=int),
                                 carry_state = (jnp.zeros((1,self.features),dtype=jVMC.util.global_defs.tReal),jnp.zeros((1,self.features),dtype=jVMC.util.global_defs.tReal),), 
                                 output_state=True)
            choice = jrnd.categorical(key[0], logits.ravel()) # abide by the indexing convention and apply -1
            _, s = self._scanning_fn((jnp.expand_dims(choice,0), carry),(key[1:],jnp.arange(1,self.L)))
            return jnp.concatenate([jnp.expand_dims(choice,0),s])

        # get the samples
        keys = jrnd.split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        logits, next_states = self(carry[0],carry_state = carry[1], output_state=True)
        choice = jrnd.categorical(key[0], logits.ravel().real) # abide by the indexing convention
        return (jnp.expand_dims(choice,0), next_states), choice

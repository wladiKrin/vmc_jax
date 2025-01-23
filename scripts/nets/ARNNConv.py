from functools import partial
from itertools import repeat, product
from typing import Any, List, Optional, Tuple

import flax
from flax.training import checkpoints
import jax
from flax.linen import (
    Dense,
    Embed,
    LayerNorm,
    Module,
    Sequential,
    compact,
    gelu,
    log_softmax,
    make_causal_mask,
    scan,
)

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from flax import linen as nn
from jax import random, numpy as jnp

from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros, roll, log, ones, pi, sin, log
from jax import Array, vmap, debug, jit
from jax.nn import log_softmax
from jax.random import categorical, split #,KeyArray

# from jVMC.global_defs import tReal
tReal = jnp.float64
class ARNNConv1D(ARNNSequential):
   """Autoregressive neural network with 1D convolution layers."""

   layers: int
   """number of layers."""
   features: tuple[int, ...] | int
   """output feature density in each layer. If a single number is given,
   all layers except the last one will have the same number of features."""
   kernel_size: int
   """length of the convolutional kernel."""
   kernel_dilation: int = 1
   """dilation factor of the convolution kernel (default: 1)."""
   activation: Callable[[Array], Array] = nkactivation.reim_selu
   """the nonlinear activation function between hidden layers (default: reim_selu)."""
   use_bias: bool = True
   """whether to add a bias to the output (default: True)."""
   param_dtype: DType = jnp.float64
   """the dtype of the computation (default: float64)."""
   precision: Any = None
   """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
   kernel_init: NNInitFunc = default_kernel_init
   """initializer for the weights."""
   bias_init: NNInitFunc = zeros
   """initializer for the biases."""
   machine_pow: int = 2
   """exponent to normalize the outputs of `__call__`."""

   def setup(self):
       features = _get_feature_list(self)
       self._layers = [
           MaskedConv1D(
               features=features[i],
               kernel_size=self.kernel_size,
               kernel_dilation=self.kernel_dilation,
               exclusive=(i == 0),
               use_bias=self.use_bias,
               param_dtype=self.param_dtype,
               precision=self.precision,
               kernel_init=self.kernel_init,
               bias_init=self.bias_init,
           )
           for i in range(self.layers)
       ]

class CpxRWKV(nn.Module):
    # Main
    L: int = 1 # system size
    LHilDim: int = 2
    patch_size: int = 1
    hidden_size: int = 8
    num_heads: int = 1
    dtype: type = tReal
    # time/channel mixing
    num_layers: int = 1
    embedding_size: int = 2
    # prob correction
    logProbFactor: float = 0.5
    # one hot embedding
    one_hot: bool = False
    # bias
    bias: bool = False
    # linear output
    lin_out: bool = False
    # sampling temperature
    temperature: float = 1.0
    # init variance
    init_variance: float = 0.1

    def setup(self):
        # set up patching
        if self.L % self.patch_size != 0:
            raise ValueError("The system size must be divisible by the patch size")
        self.patch_states = jnp.array(list(product(range(self.LHilDim),repeat=self.patch_size)))
        self.LocalHilDim = self.LHilDim ** self.patch_size
        self.PL = self.L // self.patch_size
        index_array = self.LHilDim**(jnp.arange(self.patch_size)[::-1])
        self.index_map = jax.vmap(lambda s: index_array.dot(s))
        # select type of embedding
        if self.one_hot:
            self.embed = nn.Dense(self.embedding_size, use_bias=False,name="Embedding", param_dtype=self.dtype)
        else:
            self.embed = nn.Embed(self.LocalHilDim,
                               self.embedding_size,
                               param_dtype=self.dtype)
        #self.input_layernorm = nn.LayerNorm(epsilon=1e-5,name="InputLayerNorm",param_dtype=self.dtype)
        self.blocks = [
                        RWKVBlock(layer_num=i,
                              embedding_size=self.embedding_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              bias=self.bias,
                              init_variance=self.init_variance,
                              dtype=self.dtype)
                        for i in range(self.num_layers)
                       ]
        # self.output_layernorm = nn.LayerNorm(epsilon=1e-5,name="OutputLayerNorm", param_dtype=self.dtype)
        self.neck = nn.Dense(self.hidden_size, use_bias=self.bias,name="Neck",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.dtype)
        self.head = nn.Dense(self.LocalHilDim, use_bias=False,name="Head", 
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.dtype)
        # self.PhaseNeck = nn.Dense(self.hidden_size, use_bias=self.bias,name="PhaseNeck", param_dtype=self.dtyp,
        #                         kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.dtype))
        self.PhaseHead = nn.Dense(self.LocalHilDim * self.PL, use_bias=False,name="PhaseHead",
                                kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.dtype)
        # self.PhaseHead = nn.Dense(1, use_bias=False,name="PhaseHead",
        #                         kernel_init=nn.initializers.variance_scaling(self.init_variance,mode="fan_in",distribution="truncated_normal"),param_dtype=self.dtype)

    def __call__(self, s: Array, block_states: Array = None, output_state: bool = False) -> Array:
        # the helper method allows to use nn.jit with static_argnames
        return self.forward_with_state(s, block_states, output_state)

    @partial(nn.jit, static_argnums=3)
    def forward_with_state(self, s: Array, block_states: Array = None,  output_state: bool = False) -> Array:
        
        if self.one_hot:
            if output_state:
                y = jax.nn.one_hot(s,self.LocalHilDim,axis=-1)
                y = self.embed(y)
            else:
                s = self.index_map(s.reshape(self.PL,self.patch_size))
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = jax.nn.one_hot(y,self.LocalHilDim,axis=-1) 
                y = self.embed(y)
        else:
            if output_state:
                y = self.embed(s)
            else:
                s = self.index_map(s.reshape(self.PL,self.patch_size))
                y = jnp.pad(s,(1,0),mode='constant',constant_values=0)
                y = self.embed(y)

        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            y, new_state = block(y, state)
            if output_state:
                next_states.append(new_state)
        # the neck precedes the head
        x = nn.gelu(self.neck(y))
        # the head tops the neck
        x = self.head(x)
        if self.lin_out:
            # necessery: positive activation for the probabilities with some regularization
            x = nn.elu(x) + 1. + 1e-8
            # return here for RNN mode
            if output_state:
                # normalize the product probability distribution
                x = jnp.log(x/jnp.expand_dims(x.sum(axis=-1),axis=-1))
                return x, next_states
            else:
                # normalize the product probability distribution
                x = jnp.log(x[:-1]/jnp.expand_dims(x[:-1].sum(axis=-1),axis=-1)) * self.logProbFactor
        else:
            if output_state:
                return x, next_states
            else:
                x = log_softmax(x[:-1], axis=-1) * self.logProbFactor
        # compute the phase in the auotregressive style
        # phase = nn.gelu(self.PhaseNeck(y[-1]))
        # phase = self.PhaseHead(phase)
        phase = self.PhaseHead(y[-1]).reshape(x.shape)
        # the log-probs according the state
        return (take_along_axis(x, expand_dims(s, -1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1) 
                            #    +1.j * phase.squeeze(-1) )
                + 1.j * take_along_axis(phase, expand_dims(s, -1), axis=-1)
                                .sum(axis=-2)
                                .squeeze(-1) )

    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """
        def generate_sample(key):
            key = split(key, self.PL)
            logits, carry = self(jnp.zeros(1,dtype=int),block_states = None, output_state=True)
            print(logits)
            print(carry)
            choice = categorical(key[0], logits.ravel().real/self.temperature)
            _, s = self._scanning_fn((jnp.expand_dims(choice,0),carry),key[1:])
            state = jnp.concatenate([jnp.expand_dims(choice,0),s])
            return jnp.take_along_axis(self.patch_states,state[:,None],axis=0).flatten()

        # get the samples
        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry: Tuple[Array, Array, float], key) -> Tuple[Array,Array, float]:
        logits, next_states = self(carry[0],block_states = carry[1], output_state=True)
        choice = categorical(key, logits.ravel().real/self.temperature)
        return (jnp.expand_dims(choice,0), next_states), choice

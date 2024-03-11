import jax
import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs

from functools import partial
from typing import Sequence

from jVMC.nets.initializers import init_fn_args
import jVMC.nets.initializers

class FFN_multihead(nn.Module):
    """Feed forward network with real parameters.

    Initialization arguments:
        * ``layers``: Computational basis configuration.
        * ``nHeads``: number of heads in the output
        * ``bias``: ``Boolean`` indicating whether to use bias.
        * ``actFun``: Non-linear activation function.

    """
    layers: Sequence[int] = (10,)
    nHeads: int = 6
    bias: bool = False
    actFun: Sequence[callable] = (jax.nn.elu,)

    @nn.compact
    def __call__(self, s):

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.layers) + 1):
            activationFunctions.append(self.actFun[-1])
        
        init_args = init_fn_args(dtype=global_defs.tReal, 
                                    kernel_init=jax.nn.initializers.lecun_normal(), 
                                    # kernel_init=jax.nn.initializers.zeros, 
                                    # kernel_init=jVMC.nets.initializers.new_init,
                                    # kernel_init=jax.nn.initializers.uniform(scale=1e-2),
                                    bias_init=jax.nn.initializers.constant(0))

        init_args2 = init_fn_args(dtype=global_defs.tReal, 
                                    # kernel_init=jax.nn.initializers.lecun_normal(), 
                                    # kernel_init=jax.nn.initializers.zeros, 
                                    # kernel_init=jVMC.nets.initializers.new_init,
                                    kernel_init=jax.nn.initializers.uniform(scale=1e-2),
                                    bias_init=jax.nn.initializers.zeros)

        s = 2 * s.ravel() - 1
        for l, fun in zip(self.layers, activationFunctions[:-1]):
            s = fun(
                nn.Dense(features=l, use_bias=self.bias, **init_args)(s)
            )

        return jnp.prod(
              nn.Dense(features=self.nHeads, use_bias=self.bias, **init_args)(s)
            * jnp.exp(1j * nn.Dense(features=1, use_bias=self.bias, **init_args2)(s))
        )

# ** end class FFN

class FFN_multihead_Cplx(nn.Module):
    """Feed forward network with complex parameters.

    Initialization arguments:
        * ``layers``: Computational basis configuration.
        * ``nHeads``: number of heads in the output
        * ``bias``: ``Boolean`` indicating whether to use bias.
        * ``actFun``: Non-linear activation function.

    """
    layers: Sequence[int] = (10,)
    nHeads: int = 6
    bias: bool = False
    actFun: Sequence[callable] = (jax.nn.elu,)

    @nn.compact
    def __call__(self, s):

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.layers) + 1):
            activationFunctions.append(self.actFun[-1])
        
        init_args = init_fn_args(dtype=global_defs.tCpx,
                                 kernel_init=jVMC.nets.initializers.cplx_init,
                                 bias_init=jax.nn.initializers.ones)

        s = 2 * s.ravel() - 1
        for l, fun in zip(self.layers, activationFunctions[:-1]):
            s = fun(
                nn.Dense(features=l, use_bias=self.bias, **init_args)(s)
            )

        return jnp.prod(nn.Dense(features=self.nHeads, use_bias=self.bias, **init_args)(s)
                )

# ** end class FFN

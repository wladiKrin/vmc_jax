import jax
jax.config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.util.symmetries import LatticeSymmetry

from functools import partial
from typing import List, Sequence

import jVMC.nets.initializers
from jVMC.nets.initializers import init_fn_args


class ResNet(nn.Module):
    """Convolutional neural network with residual connections with real parameters and complex output.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``bias``: Whether to use biases
    """

    F: Sequence[int] = (3,)
    channels: Sequence[int] = (16,)
    strides: Sequence[int] = (1,)
    bias: bool = True

    @nn.compact
    def __call__(self, x):

        nsites = x.size
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)

        for nblock in range(len(self.channels)):

            residual = x
            x /= np.sqrt(nblock+1, dtype=global_defs.tReal)

            if nblock == 0:
                x /= np.sqrt(2, dtype=global_defs.tReal)
            else:
                x = nn.gelu(x)

            x = nn.Conv(
                features=self.channels[nblock], 
                kernel_size=tuple(self.F), 
                padding="CIRCULAR", 
                strides=self.strides, 
                use_bias=self.bias, 
                param_dtype=global_defs.tReal, 
                dtype=global_defs.tReal, 
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init = jax.nn.initializers.zeros
                )(x)

            x = nn.gelu(x)

            x = nn.Conv(
                features=self.channels[nblock], 
                kernel_size=tuple(self.F), 
                padding="CIRCULAR", 
                strides=self.strides, 
                use_bias=self.bias and not (nblock==len(self.channels)-1), 
                param_dtype=global_defs.tReal, 
                dtype=global_defs.tReal, 
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init = jax.nn.initializers.zeros
                )(x)

            x += residual

        x /= np.sqrt(nblock+1, dtype=global_defs.tReal)
        
        x = jax.lax.complex(x[...,:(x.shape[-1]//2)], x[...,(x.shape[-1]//2):])

        x = x.astype(jnp.complex128)

        x = jax.scipy.special.logsumexp(x) - jnp.log(self.channels[nblock]*nsites)

        return x

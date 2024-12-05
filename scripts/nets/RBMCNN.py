import jax
jax.config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.util.symmetries import LatticeSymmetry

from functools import partial
from typing import List, Sequence

import jVMC.nets.initializers
from jVMC.nets.initializers import init_fn_args


class CpxRBMCNN(nn.Module):
    """Convolutional neural network with complex parameters.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer
        * ``periodicBoundary``: Whether to use periodic boundary conditions

    """
    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (jnp.cosh,)
    # actFun: Sequence[callable] = (act_funs.log_cosh,)
    bias: bool = False
    firstLayerBias: bool = False
    periodicBoundary: bool = True

    @nn.compact
    def __call__(self, x):

        #initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)
        # initFunction = jVMC.nets.initializers.cplx_variance_scaling
        initFunction = jVMC.nets.initializers.cplx_init

        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        pads = [(0, 0)]
        for f in self.F:
            if self.periodicBoundary:
                pads.append((0, f - 1))
            else:
                pads.append((f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        
        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        init_args = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction)

        # List of axes that will be summed for symmetrization
        reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        for c, f, b in zip(self.channels, activationFunctions, bias):
            if self.periodicBoundary:
                x = jnp.pad(x, pads, 'wrap')
            else:
               x = jnp.pad(x, pads, 'constant', constant_values=0)
            x = f(nn.Conv(features=c, kernel_size=tuple(self.F),
                          strides=self.strides,
                          use_bias=bias, **init_args)(x))

        return jnp.prod(x)

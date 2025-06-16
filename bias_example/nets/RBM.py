import jax

jax.config.update("jax_enable_x64", True)
from functools import partial
from typing import List, Sequence

import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
import jVMC.nets.initializers
from jVMC.nets.initializers import init_fn_args
from jVMC.util.symmetries import LatticeSymmetry


class CpxRBMNoLog(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.prod(jnp.cosh(layer(2 * s.ravel() - 1)))


class CpxRBMLog(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_variance_scaling,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))

class CpxRBMNoLog(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.prod(jnp.cosh(layer(2 * s.ravel() - 1)))

def cplx_init_ones(rng, shape, dtype):
    return jnp.ones(shape, dtype=global_defs.tReal) + 1.j * jnp.zeros(shape, dtype=global_defs.tReal)

def cplx_init2(rng, shape, dtype):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform()
    return 1000*(unif(rng1, shape, dtype=global_defs.tReal) + 1.j * unif(rng2, shape, dtype=global_defs.tReal))

class singleParamState(nn.Module):
    """
    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(
                             kernel_init=cplx_init2,
                             # kernel_init=jVMC.nets.initializers.cplx_variance_scaling,
                             # kernel_init=cplx_init_ones,
                             # kernel_init=jax.nn.initializers.ones,
                             # bias_init=jax.nn.initializers.zeros,
                             bias_init=jVMC.nets.initializers.cplx_init,
                             dtype=global_defs.tCpx)
                         )

        return jnp.prod(jnp.log(layer(2 * s.ravel() - 1)))
        # return jnp.cosh(jnp.sum(layer(2 * s.ravel() - 1)))

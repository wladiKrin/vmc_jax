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

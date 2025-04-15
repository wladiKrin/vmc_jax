import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.initializers import init_fn_args
import jVMC.nets.activation_functions as act_funs

from functools import partial

class LinkList:

    def __init__(self, links):

        self._links = links

    @property
    def links(self):
        return self._links


def angle_init(rng, shape, dtype, amp=0.0, phase=0.0):

    amps = amp * jnp.ones(shape[:-1], dtype=dtype)
    phases = phase * jnp.ones(shape[:-1], dtype=dtype)

    return jnp.stack([amps,phases], axis=2)

def angle_init_cpx(rng, shape, dtype, log_amp=0.0):

    amps = log_amp * jnp.ones(shape, dtype=dtype)

    return amps

class ZZWrapper(nn.Module):
    net: callable
    links: LinkList
    angle: float

    @nn.compact
    def __call__(self, s):

        l1 = self.links.links[:,0]
        l2 = self.links.links[:,1]
        corr = 1 + 4 * s[l1] * s[l2] - 2*s[l1] - 2*s[l2]

        return 1.j * self.angle * jnp.sum(corr) + self.net(s)
    

class BlochWrapperCpx(nn.Module):
    net: callable
    theta: float

    @nn.compact
    def __call__(self, s):

        bloch_angles = self.param('bloch_angles', partial(angle_init_cpx, dtype=global_defs.tCpx, log_amp=jnp.log( -1.j * jnp.tan(self.theta) )), s.ravel().shape)
        bloch_angles = jnp.stack([jnp.zeros_like(bloch_angles), bloch_angles])

        #sf = self.param('net_scale_factor', partial(angle_init_cpx, dtype=global_defs.tCpx, log_amp=1.0), (1,))

        return jnp.sum(bloch_angles[s, jnp.arange(bloch_angles.shape[1])]) + self.net(s)
    

class CpxRBM(nn.Module):
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
        #return jnp.sum(act_funs.poly5(layer(2 * s.ravel() - 1)))

# ** end class CpxRBM

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

from flax.linen.linear import (
  DenseGeneral,
  default_kernel_init,
  Dense,
)

from einops import rearrange


@jax.jit
@jax.vmap
def get_irpe(R):
    """
    Compute image relative positional encoding from parameter matrix R
    output r[h, i, j] = R[h, x, y], in which x = xi - xj, y = yi - yj
    """
    arange = [np.arange(i) for i in R.shape]
    idx = jnp.stack(jnp.meshgrid(*arange, indexing="ij"), axis=-1)
    idx1 = jnp.expand_dims(idx, np.arange(R.ndim, 2 * R.ndim).tolist())
    idx2 = jnp.expand_dims(idx, np.arange(R.ndim).tolist())
    diff = (idx1 - idx2).reshape(R.size * R.size, R.ndim)
    diff = tuple(diff.T)
    return R[diff].reshape(R.size, R.size)


class Transformer_block(nn.Module):
    
    h: int = 2
    c: int = 8
    d: int = 4
    norm: int = 1
    
    @nn.compact
    def __call__(self, x):
        
        norm = self.norm
        dtype = global_defs.tReal
        m, n = x.shape[1], x.shape[2] # shapes of lattice

        """******************************************
                         Convolution
        ******************************************"""
        
        residue=x
        x /= np.sqrt(norm, dtype=dtype)
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.c, 
            kernel_size=(3,3), 
            strides = (1,1),  
            padding="CIRCULAR", 
            use_bias=True, 
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=jax.nn.initializers.he_normal(),
           )(x)
        
        x+=residue
        norm+=1
        
        
        """******************************************
                       Attention
        ******************************************"""
        
        residue=x
        x /= np.sqrt(norm, dtype=dtype)
        
        q = DenseGeneral(
              axis=-1,
              dtype=dtype,
              param_dtype=dtype,
              features=self.d*self.h,
              kernel_init=jax.nn.initializers.lecun_normal(),
              use_bias=False,
              name="query",
            )(x)
        k = DenseGeneral(
              axis=-1,
              dtype=dtype,
              param_dtype=dtype,
              features=self.d*self.h,
              kernel_init=jax.nn.initializers.lecun_normal(),
              use_bias=False,
              name="key",
            )(x)
        v = DenseGeneral(
              axis=-1,
              dtype=dtype,
              param_dtype=dtype,
              features=self.d*self.h,
              kernel_init=jax.nn.initializers.lecun_normal(),
              use_bias=False,
              name="value",
            )(x)
    
        q = rearrange(q, "b x y (h d) -> b h (x y) d", h=self.h, d=self.d)
        k = rearrange(k, "b x y (h d) -> b h (x y) d", h=self.h, d=self.d)
        v = rearrange(v, "b x y (h d) -> b h (x y) d", h=self.h, d=self.d)

        p = self.param("Positional Encoding", init_fn=jax.nn.initializers.lecun_normal(), dtype=dtype, shape=(self.h, x.shape[1], x.shape[2]))
        P = get_irpe(p)
        P = jnp.expand_dims(P, 0)
        
        attention = nn.softmax((jnp.einsum('bhnd,bhmd->bhnm', q, k) + P)/jnp.sqrt(self.d))
        x = jnp.einsum('bhnn,bhnd->bhnd', attention, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        
        x = DenseGeneral(
              axis=-1,
              dtype=dtype,
              param_dtype=dtype,
              features=self.c,
              kernel_init=jax.nn.initializers.lecun_normal(),
              use_bias=False,
            )(x)

        x = rearrange(x, "b (m n) r -> b m n r", m=m, n=n)
        
        x+=residue
        norm+=1
        
        """******************************************
                Multi-layer perceptron (MLP)
        ******************************************"""
        
        residue=x
        x /= np.sqrt(norm, dtype=dtype)
        
        x = nn.Conv(
            features=4*self.c, 
            kernel_size=(1,1), 
            strides = (1,1),  
            padding="CIRCULAR", 
            use_bias=True, 
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=jax.nn.initializers.he_normal(),
            name="MLP1",
           )(x)
        
        x = nn.gelu(x)
        
        x = nn.Conv(
            features=4*self.c, 
            kernel_size=(3,3), 
            strides = (1,1),  
            feature_group_count=4*self.c, 
            padding="CIRCULAR", 
            use_bias=True, 
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=jax.nn.initializers.lecun_normal(),
            name="MLP2",
           )(x)

        x = nn.gelu(x)
        
        x = nn.Conv(
            features=self.c, 
            kernel_size=(1,1), 
            strides = (1,1),  
            padding="CIRCULAR", 
            use_bias=True, 
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=jax.nn.initializers.he_normal(),
            name="MLP3",
           )(x)

        x+=residue
        
        return x


class Convolutional_meet_vit(nn.Module):

    c: int = 8
    d: int = 16
    h: int = 2
    strides: Sequence[int] = (2,2)
    n_l: int = 2
    
    @nn.compact
    def __call__(self, x):

        nsites = x.size

        """******************************************
                       Embedding layer
        ******************************************"""
        dtype = global_defs.tReal

        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        x = nn.Conv(features=self.c, 
                    kernel_size=self.strides, 
                    strides=self.strides,  
                    padding="CIRCULAR", 
                    use_bias=False, 
                    dtype=dtype,
                    param_dtype=dtype,
                    kernel_init=jax.nn.initializers.lecun_normal(),
                    name = "Embedding Layer",
                   )(x)

        """******************************************
                   Transformer Encoder
        ******************************************"""

        norm = 1 
        for layer in range(self.n_l):
            x = Transformer_block(
                    c = self.c,
                    d = self.d,
                    h = self.h,
                    norm = norm,
                )(x)
            norm+=3
            
        """******************************************
                        Activation
        ******************************************"""

        x /= np.sqrt(norm, dtype=dtype)

        x = jax.lax.complex(x[...,:(x.shape[-1]//2)], x[...,(x.shape[-1]//2):])

        x = x.astype(jnp.complex128)

        x = jax.scipy.special.logsumexp(x) - jnp.log(self.c*nsites)
        
        return x
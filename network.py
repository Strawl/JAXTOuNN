import numpy as np

import jax.nn as nn
from jax.nn import initializers
from jax import jit, random
np.random.seed(0)

rand_key = random.PRNGKey(0) # reproducibility
from jax.example_libraries import stax

def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun
Swish = elementwise(nn.swish)
LeakyRelu = elementwise(nn.leaky_relu)

class TopNet:
  def __init__(self, nnSettings):
    self.nnSettings = nnSettings
    init_fn, applyNN = self.makeNetwork(nnSettings)
    self.fwdNN = jit(lambda nnwts, x: applyNN(nnwts, x))
    _, self.wts = init_fn(rand_key, (-1, nnSettings['inputDim']))
    
  def makeNetwork(self, nnSettings):
    # JAX network definition
    activationName = nnSettings.get('activation', 'swish').lower()
    activation = LeakyRelu if activationName in ('leaky_relu', 'leakyrelu') else Swish
    useBatchNorm = nnSettings.get('useBatchNorm', False)
    weightInit = initializers.glorot_normal()
    biasInit = initializers.zeros
    layers = []
    for i in range(nnSettings['numLayers']-1):
      layers.append(stax.Dense(nnSettings['numNeuronsPerLayer'], W_init=weightInit, b_init=biasInit))
      if(useBatchNorm):
        layers.append(stax.BatchNorm(axis=(0,)))
      layers.append(activation)
    layers.append(stax.Dense(nnSettings['outputDim'], W_init=weightInit, b_init=biasInit))
    layers.append(stax.Sigmoid)
    return stax.serial(*layers)
  
  def forward(self, wts, x):
    return 0.01 + self.fwdNN(wts, x)

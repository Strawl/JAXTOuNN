import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyDensityProjection, applyFourierMap, applySymmetry
from jax.example_libraries import optimizers

class TOuNN:
  def __init__(self, exampleName, mesh, material, nnSettings, symMap, fourierMap, densityProjection=None):
    self.exampleName = exampleName
    self.FE = JAXSolver(mesh, material)
    self.xy = jnp.array(self.FE.mesh.elemCenters)
    self.fourierMap = fourierMap
    if(fourierMap['isOn']):
      nnSettings['inputDim'] = 2*fourierMap['numTerms']
    else:
      nnSettings['inputDim'] = self.FE.mesh.ndim
    self.topNet = TopNet(nnSettings)
    
    self.symMap = symMap
    if(densityProjection is None):
      densityProjection = {'isOn':False, 'sharpness':8.}
    self.densityProjection = densityProjection
    self.domainOrigin = jnp.array([self.FE.mesh.bb['xmin'], self.FE.mesh.bb['ymin']])
    self.domainSize = jnp.array([
      self.FE.mesh.bb['xmax'] - self.FE.mesh.bb['xmin'],
      self.FE.mesh.bb['ymax'] - self.FE.mesh.bb['ymin']
    ])
    #-----------------------#

  def preprocessCoordinates(self, xy):
    xy = applySymmetry(xy, self.symMap)
    if(self.fourierMap['isOn']):
      return applyFourierMap(xy, self.fourierMap)
    return (xy - self.domainOrigin)/self.domainSize
  #-----------------------#

  def computeDensity(self, nnwts, xy):
    density = self.topNet.forward(nnwts, xy).reshape(-1)
    return applyDensityProjection(density, self.densityProjection)
  #-----------------------#
  
  def optimizeDesign(self, optParams, disableDisplay=False):
    convgHistory = {'epoch':[], 'vf':[], 'J':[]}
    xy = self.preprocessCoordinates(self.xy)

    # Original hardcoded defaults were: penal=1.0, pMax=8.0, delP=0.02
    penal = optParams.get('penal', {}).get('p0', 1.0)
    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
    opt_state = opt_init(self.topNet.wts)
    opt_update = jit(opt_update)
    self.trainedWts = get_params(opt_state)
    
    # fwd once to get J0-scaling param
    density0 = self.topNet.forward(get_params(opt_state), xy).reshape(-1)
    J0 = self.FE.objectiveHandle(density0, penal)
  
    def computeLoss(objective, constraints):
      if(optParams['lossMethod']['type'] == 'penalty'):
        alpha = optParams['lossMethod']['alpha0'] + \
                epoch*optParams['lossMethod']['delAlpha'] # penalty method
        loss = objective
        for c in constraints:
          loss += alpha*c**2
      if(optParams['lossMethod']['type'] == 'logBarrier'):
        t = optParams['lossMethod']['t0']* \
                          optParams['lossMethod']['mu']**epoch
        loss = objective
        for c in constraints:
          if(c < (-1/t**2)):
            psi = -jnp.log(-c)/t
          else:
            psi = t*c - jnp.log(1/t**2)/t + 1/t
          loss += psi
      return loss
        
    # closure function
    def closure(nnwts):
      density = self.topNet.forward(nnwts, xy).reshape(-1)
      volCons = (jnp.mean(density)/optParams['desiredVolumeFraction'])- 1.
      J = self.FE.objectiveHandle(density, penal)
      return computeLoss(J/J0, [volCons])
    
    # optimization loop
    for epoch in range(optParams['maxEpochs']):
      # Original hardcoded: min(8.0, 1. + epoch*0.02)
      pMax = optParams.get('penal', {}).get('pMax', 8.0)
      delP = optParams.get('penal', {}).get('delP', 0.02)
      penal = min(pMax, optParams.get('penal', {}).get('p0', 1.0) + epoch*delP)
      grads = jax.grad(closure)(get_params(opt_state))
      if(optParams['gradclip']['isOn']):
        grads = optimizers.clip_grads(grads, optParams['gradclip']['thresh'])
      opt_state = opt_update(epoch, grads, opt_state)
      self.trainedWts = get_params(opt_state)
  
      convgHistory['epoch'].append(epoch)
      density = self.topNet.forward(get_params(opt_state), xy).reshape(-1)

      J = self.FE.objectiveHandle(density, penal)
      convgHistory['J'].append(J)
      volf= jnp.mean(density)
      convgHistory['vf'].append(volf)
      if(epoch == 10):
        J0 = J;
      status = 'epoch {:d}, J {:.2E}, vf {:.2F}'.format(epoch, J/J0, volf);
      print(status)
      if not disableDisplay:
        if(epoch%30 == 0):
          self.FE.mesh.plotFieldOnMesh(density, status)
      else:
          self.FE.mesh.saveFieldSnapshot(density, status, epoch)
    return convgHistory

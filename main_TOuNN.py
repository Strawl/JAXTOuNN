import numpy as np
import matplotlib.pyplot as plt
from examples import getExampleBC
from Mesher import RectangularGridMesher
from projections import computeFourierMap, scaleSymmetryMap
from material import Material
from TOuNN import TOuNN
from plotUtil import plotConvergence
from export_density_vtu import dense_cell_centers, evaluate_density, write_vtu, write_density_png
from FE_Solver import JAXSolver


import configparser

#%% read config file
configFile = './config.txt'
config = configparser.ConfigParser()
config.read(configFile)

#%% Mesh and BC
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim') # default for 2
nelx = meshConfig.getint('nelx') # number of FE elements along X
nely = meshConfig.getint('nely') # number of FE elements along Y
example = meshConfig.getint('example', fallback=2)
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)
exampleName, bcSettings, symMap = getExampleBC(example, nelx, nely)
mesh = RectangularGridMesher(ndim, nelx, nely, elemSize, bcSettings)
symMap = scaleSymmetryMap(symMap, mesh)

#%% Material
materialConfig = config['MATERIAL']
E, nu =  materialConfig.getfloat('E'), materialConfig.getfloat('nu')
Emin = materialConfig.getfloat('Emin', fallback=1e-3*E)
matProp = {'physics':'structural', 'Emax':E, 'nu':nu, 'Emin':Emin}
material = Material(matProp)

#%% NN
tounnConfig = config['TOUNN']
nnSettings = {'numLayers': tounnConfig.getint('numLayers'),\
              'numNeuronsPerLayer':tounnConfig.getint('hiddenDim'),\
              'outputDim':tounnConfig.getint('outputDim'),\
              'activation':tounnConfig.get('activation', fallback='leakyrelu'),\
              'useBatchNorm':tounnConfig.getboolean('useBatchNorm', fallback=True)}
  
fourierMap = {'isOn':tounnConfig.getboolean('fourier_isOn'),\
              'minRadius':tounnConfig.getfloat('fourier_minRadius'), \
              'maxRadius':tounnConfig.getfloat('fourier_maxRadius'),\
              'numTerms':tounnConfig.getint('fourier_numTerms')}

fourierMap['map'] = computeFourierMap(mesh, fourierMap)

#%% Optimization params
lossConfig = config['LOSS']
lossType = lossConfig.get('type', fallback='logBarrier')
lossMethod = {'type':lossType, 't0':lossConfig.getfloat('t0'),\
              'mu':lossConfig.getfloat('mu'),\
              'alpha0':lossConfig.getfloat('alpha0'),\
              'delAlpha':lossConfig.getfloat('delAlpha'),\
              'alphaMax':lossConfig.getfloat('alphaMax', fallback=np.inf)}
          
optConfig = config['OPTIMIZATION']
optParams = {'maxEpochs':optConfig.getint('numEpochs'),\
             'lossMethod':lossMethod,\
             'learningRate':optConfig.getfloat('lr'),\
             'desiredVolumeFraction':optConfig.getfloat('desiredVolumeFraction'),\
             'penal':{'p0':optConfig.getfloat('p0', fallback=1.),\
                      'pMax':optConfig.getfloat('pMax', fallback=8.),\
                      'delP':optConfig.getfloat('delP', fallback=0.02)},\
             'gradclip':{'isOn':optConfig.getboolean('gradClip_isOn'),\
                         'thresh':optConfig.getfloat('gradClip_clipNorm')}}

densityProjection = {'isOn':False, 'sharpness':8.}
if(config.has_section('DENSITY_PROJECTION')):
  projectionConfig = config['DENSITY_PROJECTION']
  densityProjection = {'isOn':projectionConfig.getboolean('isOn', fallback=False),\
                       'sharpness':projectionConfig.getfloat('sharpness', fallback=8.)}

#%% Run optimization
plt.close('all')
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, densityProjection)

# Check if live display should be disabled (save snapshots instead)
disableDisplay = tounnConfig.getboolean('disableDisplay', fallback=False)

convgHistory = tounn.optimizeDesign(optParams, disableDisplay=disableDisplay)
plotConvergence(convgHistory)

if(config.has_section('EXPORT') and config['EXPORT'].getboolean('enabled', fallback=False)):
  exportConfig = config['EXPORT']
  exportRes = exportConfig.getint('res', fallback=3)
  exportPath = exportConfig.get('output', fallback='results/tounn_density_res3.vtu')
  pngPath = exportConfig.get('output_png', fallback='results/tounn_density_res3.png')
  xyDense = dense_cell_centers(mesh, exportRes)
  densityDense = evaluate_density(tounn, xyDense)
  exportedVolumeFraction = float(np.mean(densityDense))
  write_vtu(exportPath, mesh, exportRes, densityDense)
  write_density_png(pngPath, mesh, exportRes, densityDense)
  print('Wrote {:s}'.format(exportPath))
  print('Wrote {:s}'.format(pngPath))
  print('VTU cells: {:d} x {:d}'.format(nelx*exportRes, nely*exportRes))
  print('VTU domain: {:.6g} x {:.6g}'.format(mesh.bb['xmax'], mesh.bb['ymax']))
  print('Exported volume fraction: {:.6g}'.format(exportedVolumeFraction))

  # Evaluate compliance on the exported-resolution mesh using the exported density.
  fineNelx, fineNely = nelx*exportRes, nely*exportRes
  _, fineBcSettings, _ = getExampleBC(example, fineNelx, fineNely)
  # `elemSize` is historically named and actually stores element density
  # (elements per physical unit). Scale it with exportRes so the refined mesh
  # keeps the same physical domain size instead of expanding it.
  fineElemSize = elemSize*exportRes
  fineMesh = RectangularGridMesher(ndim, fineNelx, fineNely, fineElemSize, fineBcSettings)
  fineFE = JAXSolver(fineMesh, material)
  finalPenal = min(
      optParams['penal']['pMax'],
      optParams['penal']['p0'] + (optParams['maxEpochs'] - 1)*optParams['penal']['delP']
  )
  exportedCompliance = fineFE.objectiveHandle(densityDense, finalPenal)
  print('Exported-resolution compliance: {:.6g}'.format(exportedCompliance))

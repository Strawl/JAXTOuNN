import argparse
import configparser
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from examples import getExampleBC
from material import Material
from Mesher import RectangularGridMesher
from projections import computeFourierMap, scaleSymmetryMap
from TOuNN import TOuNN


def build_problem(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    mesh_config = config["MESH"]
    ndim = mesh_config.getint("ndim")
    nelx = mesh_config.getint("nelx")
    nely = mesh_config.getint("nely")
    example = mesh_config.getint("example", fallback=2)
    elem_size = np.array(mesh_config["elemSize"].split(",")).astype(float)
    example_name, bc_settings, sym_map = getExampleBC(example, nelx, nely)
    mesh = RectangularGridMesher(ndim, nelx, nely, elem_size, bc_settings)
    sym_map = scaleSymmetryMap(sym_map, mesh)

    material_config = config["MATERIAL"]
    emax = material_config.getfloat("E")
    nu = material_config.getfloat("nu")
    emin = material_config.getfloat("Emin", fallback=1e-3 * emax)
    material = Material(
        {"physics": "structural", "Emax": emax, "nu": nu, "Emin": emin}
    )

    tounn_config = config["TOUNN"]
    nn_settings = {
        "numLayers": tounn_config.getint("numLayers"),
        "numNeuronsPerLayer": tounn_config.getint("hiddenDim"),
        "outputDim": tounn_config.getint("outputDim"),
        "activation": tounn_config.get("activation", fallback="leakyrelu"),
        "useBatchNorm": tounn_config.getboolean("useBatchNorm", fallback=True),
    }

    fourier_map = {
        "isOn": tounn_config.getboolean("fourier_isOn"),
        "minRadius": tounn_config.getfloat("fourier_minRadius"),
        "maxRadius": tounn_config.getfloat("fourier_maxRadius"),
        "numTerms": tounn_config.getint("fourier_numTerms"),
    }
    fourier_map["map"] = computeFourierMap(mesh, fourier_map)

    loss_config = config["LOSS"]
    loss_method = {
        "type": loss_config.get("type", fallback="logBarrier"),
        "t0": loss_config.getfloat("t0"),
        "mu": loss_config.getfloat("mu"),
        "alpha0": loss_config.getfloat("alpha0"),
        "delAlpha": loss_config.getfloat("delAlpha"),
        "alphaMax": loss_config.getfloat("alphaMax", fallback=np.inf),
    }

    opt_config = config["OPTIMIZATION"]
    opt_params = {
        "maxEpochs": opt_config.getint("numEpochs"),
        "lossMethod": loss_method,
        "learningRate": opt_config.getfloat("lr"),
        "desiredVolumeFraction": opt_config.getfloat("desiredVolumeFraction"),
        "penal": {
            "p0": opt_config.getfloat("p0", fallback=1.0),
            "pMax": opt_config.getfloat("pMax", fallback=8.0),
            "delP": opt_config.getfloat("delP", fallback=0.02),
        },
        "gradclip": {
            "isOn": opt_config.getboolean("gradClip_isOn"),
            "thresh": opt_config.getfloat("gradClip_clipNorm"),
        },
    }

    density_projection = {"isOn": False, "sharpness": 8.0}
    if config.has_section("DENSITY_PROJECTION"):
        projection_config = config["DENSITY_PROJECTION"]
        density_projection = {
            "isOn": projection_config.getboolean("isOn", fallback=False),
            "sharpness": projection_config.getfloat("sharpness", fallback=8.0),
        }

    tounn = TOuNN(
        example_name,
        mesh,
        material,
        nn_settings,
        sym_map,
        fourier_map,
        density_projection,
    )
    return tounn, opt_params


def dense_cell_centers(mesh, res):
    nx = mesh.nelx * res
    ny = mesh.nely * res
    dx = mesh.elemSize[0] / res
    dy = mesh.elemSize[1] / res

    xy = np.zeros((nx * ny, 2))
    index = 0
    for i in range(nx):
        for j in range(ny):
            xy[index, 0] = (i + 0.5) * dx
            xy[index, 1] = (j + 0.5) * dy
            index += 1
    return xy


def evaluate_density(tounn, xy):
    xy = tounn.preprocessCoordinates(jnp.array(xy))
    density = tounn.computeDensity(tounn.trainedWts, xy)
    return np.array(density).reshape(-1)


def write_vtu(path, mesh, res, density):
    nx = mesh.nelx * res
    ny = mesh.nely * res
    dx = mesh.elemSize[0] / res
    dy = mesh.elemSize[1] / res

    points = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            points.append((i * dx, j * dy, 0.0))

    connectivity = []
    offsets = []
    cell_types = []
    offset = 0
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            connectivity.extend([n0, n1, n2, n3])
            offset += 4
            offsets.append(offset)
            cell_types.append(9)  # VTK_QUAD

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{nx * ny}">\n')

        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y, z in points:
            f.write(f"          {x:.16g} {y:.16g} {z:.16g}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(map(str, connectivity)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(map(str, offsets)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(map(str, cell_types)) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")

        f.write('      <CellData Scalars="density">\n')
        f.write('        <DataArray type="Float64" Name="density" format="ascii">\n')
        for value in density:
            f.write(f"          {float(value):.16g}\n")
        f.write("        </DataArray>\n")
        f.write("      </CellData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train TOuNN from config.txt and export a dense density VTU."
    )
    parser.add_argument("--config", default="config.txt")
    parser.add_argument("--res", type=int, default=3)
    parser.add_argument("--output", default="results/tounn_density_res3.vtu")
    args = parser.parse_args()

    if args.res < 1:
        raise ValueError("--res must be >= 1")

    tounn, opt_params = build_problem(args.config)
    tounn.optimizeDesign(opt_params)

    xy = dense_cell_centers(tounn.FE.mesh, args.res)
    density = evaluate_density(tounn, xy)
    write_vtu(args.output, tounn.FE.mesh, args.res, density)
    print(f"Wrote {args.output}")
    print(
        f"VTU cells: {tounn.FE.mesh.nelx * args.res} x "
        f"{tounn.FE.mesh.nely * args.res}"
    )
    print(
        "VTU domain: "
        f"{tounn.FE.mesh.bb['xmax']:.6g} x {tounn.FE.mesh.bb['ymax']:.6g}"
    )
    print(
        "density min/mean/max: "
        f"{density.min():.6g} / {density.mean():.6g} / {density.max():.6g}"
    )


if __name__ == "__main__":
    main()

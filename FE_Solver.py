import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

from feax_cholmod_solver import cholmod_solve

class JAXSolver:
  def __init__(self, mesh, material):
    self.mesh = mesh
    self.material = material
    self.D0 = jnp.array(self.material.getD0elemMatrix(self.mesh))
    self._setup_sparse_free_dof_assembly()
    self._objective_impl = self._make_objective()
    self.objectiveHandle = jit(self._objective_impl)
  #-----------------------# 
  def _setup_sparse_free_dof_assembly(self):
    free = np.asarray(self.mesh.bc['free'], dtype=np.int64)
    free_map = -np.ones(self.mesh.ndof, dtype=np.int64)
    free_map[free] = np.arange(free.size, dtype=np.int64)

    row_full = np.asarray(self.mesh.nodeIdx[0], dtype=np.int64)
    col_full = np.asarray(self.mesh.nodeIdx[1], dtype=np.int64)
    free_entry_mask = (free_map[row_full] >= 0) & (free_map[col_full] >= 0)

    row = free_map[row_full[free_entry_mask]]
    col = free_map[col_full[free_entry_mask]]
    n_free = free.size
    pair_codes = row*n_free + col
    unique_codes, inverse = np.unique(pair_codes, return_inverse=True)
    csr_rows = unique_codes//n_free
    csr_columns = unique_codes % n_free
    row_counts = np.bincount(csr_rows, minlength=n_free)
    csr_offsets = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    self._free = jnp.array(free, dtype=jnp.int32)
    self._edof_mat = jnp.array(self.mesh.edofMat, dtype=jnp.int32)
    self._force_free = jnp.array(self.mesh.bc['force'][free].reshape(-1))
    self._free_entry_idx = jnp.array(np.flatnonzero(free_entry_mask), dtype=jnp.int32)
    self._csr_inverse = jnp.array(inverse, dtype=jnp.int32)
    self._csr_columns = jnp.array(csr_columns.astype(np.int32))
    self._csr_offsets = jnp.array(csr_offsets)
    self._nnz = unique_codes.size
    self._cache_namespace = 'tounn_{:x}'.format(id(self))
  #-----------------------#
  def _youngs_modulus(self, density, penal):
    return self.material.matProp['Emin'] + \
          (self.material.matProp['Emax']-self.material.matProp['Emin'])*\
            density**penal
  #-----------------------#
  def _assemble_free_csr_values(self, Y):
    sK = jnp.einsum('e,jk->ejk', Y, self.D0).flatten()
    sK_free = sK[self._free_entry_idx]
    return jnp.zeros((self._nnz,), dtype=sK.dtype).at[self._csr_inverse].add(sK_free)
  #-----------------------#
  def _solve_displacement(self, density, penal):
    Y = self._youngs_modulus(density, penal)
    csr_values = self._assemble_free_csr_values(Y)
    u_free = cholmod_solve(
        self._force_free,
        csr_values,
        self._csr_offsets,
        self._csr_columns,
        lower=False,
        order='amd',
        cache_namespace=self._cache_namespace,
    )
    u = jnp.zeros((self.mesh.ndof,), dtype=u_free.dtype)
    u = u.at[self._free].set(u_free) # homogeneous Dirichlet values on fixed dofs
    return jnp.dot(self._force_free, u_free), u
  #-----------------------#
  def _make_objective(self):
    @jax.custom_vjp
    def objective(density, penal):
      J, _ = self._solve_displacement(density, penal)
      return J

    def objective_fwd(density, penal):
      J, u = self._solve_displacement(density, penal)
      return J, (density, penal, u)

    def objective_bwd(res, g):
      density, penal, u = res
      elem_u = u[self._edof_mat]
      dJ_dY = -jnp.einsum('ei,ij,ej->e', elem_u, self.D0, elem_u)
      dY_ddensity = (self.material.matProp['Emax']-self.material.matProp['Emin'])*\
            penal*density**(penal-1.)
      return (g*dJ_dY*dY_ddensity, jnp.zeros_like(penal))

    objective.defvjp(objective_fwd, objective_bwd)
    return objective
  #-----------------------#
  def objective(self, density, penal):
    return self._objective_impl(density, penal)

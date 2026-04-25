"""FEAX-derived CHOLMOD sparse solve callback for TOuNN.

This keeps the linear solve interface small: solve one CSR SPD system
``A x = b`` from inside JAX via ``jax.pure_callback``.  CHOLMOD is used when
``scikit-sparse`` is installed; SciPy's sparse direct solver is retained as a
portable fallback so the project remains runnable without SuiteSparse.
"""

import functools as ft

import jax
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _build_csr_matrix(values, columns, offsets, n):
  nnz = int(offsets[-1])
  return sp.csr_matrix((values[:nnz], columns[:nnz], offsets), shape=(n, n))


def _make_solver_state():
  return {
      'factor': None,
      'csr_columns': None,
      'csr_offsets': None,
      'n': None,
      'backend': None,
  }


def _factorize_and_solve_with_cache(state, *, lower, order, values, columns, offsets, rhs):
  n = int(rhs.shape[0])
  A_csr = _build_csr_matrix(values, columns, offsets, n)

  pattern_changed = (
      state['factor'] is None
      or state['n'] != n
      or state['csr_columns'] is None
      or state['csr_columns'].shape != columns.shape
      or not np.array_equal(state['csr_columns'], columns)
      or state['csr_offsets'] is None
      or state['csr_offsets'].shape != offsets.shape
      or not np.array_equal(state['csr_offsets'], offsets)
  )

  try:
    from sksparse import cholmod
  except ImportError:
    state['backend'] = 'scipy'
    return np.asarray(spla.spsolve(A_csr, rhs), dtype=rhs.dtype)

  A_csc = A_csr.tocsc()
  if pattern_changed or state['backend'] != 'cholmod':
    state['factor'] = cholmod.cho_factor(A_csc, lower=lower, order=order, sym_kind='sym')
    state['csr_columns'] = columns.copy()
    state['csr_offsets'] = offsets.copy()
    state['n'] = n
    state['backend'] = 'cholmod'
  else:
    state['factor'].factorize(A_csc)

  return np.asarray(state['factor'].solve(rhs), dtype=rhs.dtype)


@ft.lru_cache(maxsize=None)
def _get_cholmod_host_callback(lower: bool, order: str, cache_namespace: str):
  del cache_namespace
  state = _make_solver_state()

  def _host_callback(csr_values, csr_columns, csr_offsets, b_values):
    values_np = np.asarray(csr_values)
    columns_np = np.asarray(csr_columns)
    offsets_np = np.asarray(csr_offsets)
    b_np = np.array(b_values, copy=True)

    return _factorize_and_solve_with_cache(
        state,
        lower=lower,
        order=order,
        values=values_np,
        columns=columns_np,
        offsets=offsets_np,
        rhs=b_np,
    )

  return _host_callback


@ft.partial(jax.jit, static_argnames=('lower', 'order', 'cache_namespace'))
def cholmod_solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    lower: bool = False,
    order: str = 'amd',
    cache_namespace: str = 'global',
):
  """Solve one SPD sparse system in CSR format with CHOLMOD when available."""
  if b_values.ndim != 1:
    raise ValueError('b_values must have shape (n,).')
  if csr_offsets.shape[0] != b_values.shape[0] + 1:
    raise ValueError('csr_offsets must have length n + 1.')
  if csr_values.ndim != 1 or csr_columns.ndim != 1:
    raise ValueError('csr_values and csr_columns must be rank-1.')
  if csr_values.shape[0] != csr_columns.shape[0]:
    raise ValueError('csr_values length must match csr_columns length.')

  callback = _get_cholmod_host_callback(lower, order, cache_namespace)
  result_shape = jax.ShapeDtypeStruct(b_values.shape, b_values.dtype)
  return jax.pure_callback(
      callback,
      result_shape,
      csr_values,
      csr_columns,
      csr_offsets,
      b_values,
  )

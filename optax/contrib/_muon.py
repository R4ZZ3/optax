# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2:
    raise ValueError(f'Input must have shape (m, n), got {x.shape}')
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )
  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x
  transposed = False
  if x.shape[0] > x.shape[1]:
    x = x.T
    transposed = True
  x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
  ns_coeffs = ns_coeffs.astype(x.dtype)
  if ns_coeffs.ndim == 1:
    x = jax.lax.fori_loop(
        0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
    )
  else:
    x, _ = jax.lax.scan(
        lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
    )
  if transposed:
    x = x.T
  return x


class MuonState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates


def scale_by_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.1,
    adaptive: bool = False,
    eps: float = 1e-8,
) -> base.GradientTransformation:
  """Scale updates using the Muon optimizer.

  Muon (MomentUm Orthogonalized by Newton-schulz) internally runs standard
  SGD-momentum, and then performs an orthogonalization post-processing step,
  in which each 2D parameter's update is replaced with the nearest orthogonal
  matrix. To efficiently orthogonalize each update, we use a Newton-Schulz
  iteration.

  References:
    [Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
    by Keller Jordan

  Args:
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (3,) or (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    momentum: Momentum parameter.
    nesterov: Whether to use Nesterov momentum.
    weight_decay: Weight decay parameter.
    adaptive: Whether to scale updates by the dual norm.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """
  def init_fn(params):
    return MuonState(count=jnp.zeros([], jnp.int32), mu=otu.tree_zeros_like(params))

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(NO_PARAMS_MSG)

    # Apply momentum
    mu = otu.tree_mul(state.mu, momentum)
    mu = otu.tree_add(mu, updates)
    
    if nesterov:
      updates_with_momentum = otu.tree_add(updates, otu.tree_mul(mu, momentum))
    else:
      updates_with_momentum = mu

    # Apply weight decay
    if weight_decay > 0:
      params = otu.tree_scalar_mul(1 - weight_decay, params)

    # Process updates
    def process_param_update(update, param):
      if update.ndim != 2:
        return update
      
      # Adjust learning rate based on matrix dimensions
      A, B = param.shape
      adjusted_ratio = 0.2 * jnp.sqrt(jnp.maximum(A, B))
      
      # Orthogonalize via Newton-Schulz
      orthogonalized = orthogonalize_via_newton_schulz(
          update, ns_coeffs, ns_steps, eps)
      
      # Scale by adjusted learning rate
      return orthogonalized * adjusted_ratio

    # Apply orthogonalization to 2D parameters only
    processed_updates = jax.tree.map(
        process_param_update, updates_with_momentum, params)

    # Scale by dual norm if adaptive
    if adaptive:
      updates_norm = otu.tree_l2_norm(processed_updates)
      processed_updates = jax.tree.map(
          lambda u: u / (updates_norm + eps), processed_updates)

    return processed_updates, MuonState(
        count=numerics.safe_int32_increment(state.count), mu=mu)

  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Note that Muon is currently only defined for 2D parameters, i.e. matrices.
  This is because the Newton-Schulz iterator expects a matrix as input.
  The non-2D parameters are instead passed through an Adam optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  return combine.multi_transform(
      transforms={
          'muon': combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
              ),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
              learning_rate=learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=lambda params: jax.tree.map(
          lambda x: 'muon' if x.ndim == 2 else 'adam', params
      ),
  )

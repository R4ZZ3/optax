# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""An implementation of the Splus optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils as otu


class SPlusState(NamedTuple):
  """State for the Splus optimizer."""

  ema: chex.Array
  momentum: chex.Array
  sides: chex.Array
  q_sides: chex.Array
  step: chex.Array
  ema_rate: float


def splus_get_eval_params(state: SPlusState) -> chex.Array:
  """Get parameters for evaluation.

  This function can be used to retrieve the exponential moving average of the
  parameters, which can be useful for evaluation.

  Args:
    state: The Splus state.

  Returns:
    The parameters for evaluation (the EMA of the parameters).
  """
  ema_hat = jax.tree_map(
      lambda e: e / (1 - state.ema_rate**state.step), state.ema
  )
  return ema_hat


def splus(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    ema_rate: float = 0.999,
    eps: float = 1e-30,
    inverse_every: int = 100,
    nonstandard_constant: float = 0.001,
    nonstandard_strings: tuple[str, ...] = ('embed', 'layernorm'),
    weight_decay: float = 1e-2,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    jit_broadcast_computation: bool = False,
    jit_original_sharding: Optional[SPlusState] = None,
    max_dim: int = 10000,
) -> base.GradientTransformation:
  """A second-order-like optimizer that uses eigendecomposition.

  Splus is a second-order-like optimizer that uses an online eigendecomposition
  of the outer product of gradients to approximate the Hessian. The gradients
  are then rotated into the eigenbasis of this Hessian approximation before
  an update is applied.

  For more details, see the original implementation at:
  https://github.com/google-research/google-research/blob/master/splus/splus/experiments/optimizers/splus.py

  Args:
    learning_rate: The learning rate.
    b1: The exponential decay rate for the first moment estimates.
    b2: The exponential decay rate for the second moment estimates.
    ema_rate: The exponential decay rate for the EMA of the parameters.
    eps: A small constant for numerical stability.
    inverse_every: The frequency at which to update the eigendecomposition.
    nonstandard_constant: A constant for non-standard layers.
    nonstandard_strings: A list of strings to identify non-standard layers.
    weight_decay: The weight decay rate.
    mask: A mask to apply to the gradients.
    jit_broadcast_computation: Whether to use JIT broadcast computation.
    jit_original_sharding: The original sharding of the Splus state.
    max_dim: The maximum dimension for which to compute the eigendecomposition.

  Returns:
    A `GradientTransformation`.
  """

  def init_fn(params):
    momentum = otu.tree_zeros_like(params)
    ema = otu.tree_zeros_like(params)

    def sides_decomp(p):
      if len(p.shape) == 2:
        return [jnp.zeros((d, d)) if d < max_dim else None for d in p.shape]
      return None

    sides = jax.tree_map(sides_decomp, params)

    def qs_decomp(p):
      if len(p.shape) == 2:
        return [jnp.eye(d) if d < max_dim else None for d in p.shape]
      return None

    q_sides = jax.tree_map(qs_decomp, params)
    step = jnp.zeros([], jnp.int32)
    return SPlusState(ema, momentum, sides, q_sides, step, ema_rate)

  def update_sides(g, s):
    if len(g.shape) == 2:
      return [
          b2 * s[0] + (1 - b2) * g @ g.T if s[0] is not None else None,
          b2 * s[1] + (1 - b2) * g.T @ g if s[1] is not None else None,
      ]
    else:
      return None

  def rot(p, q):
    if len(p.shape) == 2:
      p = q[0].T @ p if q[0] is not None else p
      p = p @ q[1] if q[1] is not None else p
    return p

  def unrot(p, q):
    if len(p.shape) == 2:
      p = q[0] @ p if q[0] is not None else p
      p = p @ q[1].T if q[1] is not None else p
    return p

  @jax.jit
  def get_eigvecs(s):
    if s is None:
      return None
    _, q = jnp.linalg.eigh(s + eps * jnp.eye(s.shape[0]))
    return q

  def update_inverse(sides):
    if jit_broadcast_computation:
      devices = jax.local_devices()
      tensor_shapes = {}

      def put_device_staggered(p):
        idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
        tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
        return jax.device_put(p, devices[idx])

      sides = jax.experimental.multihost_utils.process_allgather(sides)
      sides = jax.tree_map(put_device_staggered, sides)
    q_sides = jax.tree_map(get_eigvecs, sides)
    if jit_broadcast_computation:
      q_sides = jax.tree_map(lambda _, x: jax.device_get(x), sides, q_sides)
      if jit_original_sharding is not None:
        q_sides = jax.jit(
            lambda x: x, out_shardings=jit_original_sharding.q_sides
        )(q_sides)
    return q_sides

  def update_fn(grads, state, params):
    step = state.step + 1

    if params is None:
      raise ValueError(
          'Splus optimizer requires params to be passed to the update function.'
      )

    # Rotate to eigenbasis, take sign, unrotate.
    momentum = jax.tree_map(
        lambda m, g: b1 * m + (1 - b1) * g, state.momentum, grads
    )
    momentum_rot = jax.tree_map(rot, momentum, state.q_sides)
    updates_rot = jax.tree_map(jnp.sign, momentum_rot)
    updates = jax.tree_map(unrot, updates_rot, state.q_sides)
    sides = jax.tree_map(update_sides, grads, state.sides)
    ema = jax.tree_map(
        lambda e, p: ema_rate * e + (1 - ema_rate) * p, state.ema, params
    )

    # Every `inverse_every` steps, we update the inverse eigendecomposition.
    do_inverse = (step % inverse_every == 0) | (step == 1)
    q_sides = jax.lax.cond(
        do_inverse, update_inverse, lambda _: state.q_sides, sides
    )

    return updates, SPlusState(
        ema, momentum, sides, q_sides, step, state.ema_rate
    )

  def shape_scaling(updates, state, params):
    del state, params

    def shape_scale(path, u):
      path_str = '/'.join([p.key for p in path])
      if (
          len(u.shape) == 2
          and not any(k in path_str.lower() for k in nonstandard_strings)
          and u.shape[0] < max_dim
          and u.shape[1] < max_dim
      ):
        scale = 2 / (u.shape[0] + u.shape[1])
      else:
        scale = nonstandard_constant
      return u * scale

    return jax.tree_util.tree_map_with_path(shape_scale, updates), None

  splus_main = base.GradientTransformation(init_fn, update_fn)
  splus_scaling = base.GradientTransformation(lambda _: None, shape_scaling)
  return combine.chain(
      splus_main,
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
      splus_scaling,
  ) 
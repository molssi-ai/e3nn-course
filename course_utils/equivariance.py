"""Numerical equivariance tests used throughout the course.

A function ``f`` with input irreps ``irreps_in`` and output irreps ``irreps_out`` is
O(3)-equivariant iff for every group element ``g``:

    f(D_in(g) x) = D_out(g) f(x)

where ``D_in``/``D_out`` are the (block-diagonal) representation matrices of the irreps.
These helpers draw random rotations (and optionally the inversion) and report the maximal
deviation between the two sides.
"""

from __future__ import annotations

import torch
from e3nn import o3


def _random_group_element(parity: bool, device, dtype) -> torch.Tensor:
    """A random O(3) matrix: a uniform rotation, times -1 (inversion) half the time."""
    R = o3.rand_matrix(device=device, dtype=dtype)
    if parity and torch.rand(()) < 0.5:
        R = -R
    return R


def equivariance_error(
    func,
    irreps_in: str | o3.Irreps,
    irreps_out: str | o3.Irreps,
    n_trials: int = 5,
    parity: bool = True,
    batch: int = 16,
    device=None,
    dtype=torch.float64,
) -> float:
    """Max |f(D_in x) - D_out f(x)| over random inputs and random group elements.

    ``func`` must map ``(batch, irreps_in.dim)`` -> ``(batch, irreps_out.dim)``.
    Use float64 to separate genuine equivariance breaking from round-off noise.
    """
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    err = 0.0
    for _ in range(n_trials):
        x = irreps_in.randn(batch, -1, device=device, dtype=dtype)
        g = _random_group_element(parity, device, dtype)
        D_in = irreps_in.D_from_matrix(g)
        D_out = irreps_out.D_from_matrix(g)
        lhs = func(x @ D_in.T)
        rhs = func(x) @ D_out.T
        err = max(err, (lhs - rhs).abs().max().item())
    return err


def assert_equivariant(
    func,
    irreps_in: str | o3.Irreps,
    irreps_out: str | o3.Irreps,
    tol: float = 1e-9,
    **kwargs,
) -> float:
    """Assert equivariance of ``func`` (see ``equivariance_error``); returns the error."""
    err = equivariance_error(func, irreps_in, irreps_out, **kwargs)
    assert err < tol, f"equivariance violated: max error {err:.3e} >= tol {tol:.1e}"
    print(f"equivariant!  (max error {err:.3e} over random O(3) elements)")
    return err


def model_equivariance_error(
    model,
    pos: torch.Tensor,
    irreps_out: str | o3.Irreps,
    n_trials: int = 5,
    parity: bool = True,
    translation: bool = True,
    **model_kwargs,
) -> float:
    """Equivariance error for a model taking *positions* ``(N, 3)`` (plus fixed kwargs).

    Checks  model(R pos + t) = D_out(R) model(pos)  for random rotations R (optionally
    improper, i.e. including inversion) and random translations t. Extra inputs that do
    not transform (e.g. atomic numbers, edge indices) are passed through ``model_kwargs``.
    """
    irreps_out = o3.Irreps(irreps_out)
    dtype, device = pos.dtype, pos.device
    err = 0.0
    for _ in range(n_trials):
        g = _random_group_element(parity, device, dtype)
        t = torch.randn(3, device=device, dtype=dtype) if translation else 0.0
        D_out = irreps_out.D_from_matrix(g)
        lhs = model(pos @ g.T + t, **model_kwargs)
        rhs = model(pos, **model_kwargs) @ D_out.T
        err = max(err, (lhs - rhs).abs().max().item())
    return err


def assert_model_equivariant(model, pos, irreps_out, tol: float = 1e-9, **kwargs) -> float:
    """Assert-flavored wrapper around ``model_equivariance_error``."""
    err = model_equivariance_error(model, pos, irreps_out, **kwargs)
    assert err < tol, f"equivariance violated: max error {err:.3e} >= tol {tol:.1e}"
    rot = "O(3)" if kwargs.get("parity", True) else "SO(3)"
    group = f"{rot} + translations" if kwargs.get("translation", True) else rot
    print(f"equivariant!  (max error {err:.3e} over random {group} elements)")
    return err

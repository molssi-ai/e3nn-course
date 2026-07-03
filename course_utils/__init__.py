"""Shared helpers for the e3nn equivariant-GNN course notebooks.

Modules
-------
equivariance : numerical equivariance testing (rotations + parity)
plotting     : spherical harmonics surfaces, tensor-product path diagrams, training curves
data         : toy datasets (tetris, LJ argon), neighbor lists / radius graphs
"""

from . import data, equivariance, plotting

__all__ = ["equivariance", "plotting", "data"]

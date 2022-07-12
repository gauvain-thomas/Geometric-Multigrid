# Geometric Multigrid


This repository provides a Python/PyTorch implementation of a geometric multigrid (MG) solver for elliptic equations, such as Poisson and Helmholtz-like equations.

This tool currently supports vertex-centered grid with Dirichlet boundary conditions, and cell-centered grid with Neumann boundary conditions.

## Example Usage

```Python
import torch
from solver import MG
mg = MG(dx=1, dy=1, nx=257, ny=257, nl=1, nlevels=8,
        grid_type='vertex-centered', boundary_cond='dirichlet') # Initialize the solver

rhs = torch.normal(0., 1., (1, 257,257)) # Create a random rhs
u, res = mg.solve(rhs) # Solve Δu = rhs

print(f'Residue norm : {torch.norm(res):.2E}')
```

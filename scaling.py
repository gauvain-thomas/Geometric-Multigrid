import torch
import torch.nn.functional as F

def restrict(v, mode="FW"):
    """Restrict v to the coarse grid : Dirichlet odd-sized"""
    nx, ny = v.shape[-2:]

    v_c = torch.zeros(v.shape[:-2]+(torch.div(nx, 2, rounding_mode='floor') + 1, torch.div(ny, 2, rounding_mode='floor') + 1), dtype=v.dtype, device=v.device)

    # Borders not calculated due to Dirichlet boundary conditions
    if mode=="HW":
        # Half Weighting operator
        # Interior cell points
        v_c[...,1:-1, 1:-1] = (4*v[...,2:-2:2, 2:-2:2] + v[...,1:-3:2, 2:-2:2] + v[...,2:-2:2, 1:-3:2] + v[...,3:-1:2, 2:-2:2] + v[...,2:-2:2, 3:-1:2])/8

    if mode=="FW":
        # Full Weighting Operator
        v_c[...,1:-1, 1:-1] = (4*v[...,2:-2:2, 2:-2:2]
                            + 2*v[...,1:-3:2, 2:-2:2] + 2*v[...,2:-2:2, 1:-3:2] + 2*v[...,3:-1:2, 2:-2:2] + 2*v[...,2:-2:2, 3:-1:2]
                            + v[...,1:-3:2, 1:-3:2] + v[...,1:-3:2, 3:-1:2] + v[...,3:-1:2, 1:-3:2] + v[...,3:-1:2, 3:-1:2])/16

    return v_c

def restrictN(v):
    """Restrict v to the coarse grid : Neumann even-sized"""
    nx, ny = v.shape[-2:]

    v_c = torch.zeros(v.shape[:-2]+(torch.div(nx, 2, rounding_mode='floor'), torch.div(ny, 2, rounding_mode='floor')), dtype=v.dtype, device=v.device)

    v_c[...,:, :] = (v[..., :-1:2, :-1:2] + v[..., 1::2, :-1:2] + v[..., ::2, 1::2] + v[..., 1::2, 1::2])/4

    return v_c


def interpolate(v):
    """Interpolate 'v' to the fine grid : Dirichlet odd-sized"""

    nx, ny = v.shape[-2:]
    v_f = torch.zeros(v.shape[:-2] + (2*nx-1,2*ny-1), dtype=v.dtype, device=v.device)

    v_f[..., ::2   , ::2   ] = v
    v_f[..., 1:-1:2, ::2   ] = (v[...,:-1, :] + v[...,1:, :])/2
    v_f[..., ::2   , 1:-1:2] = (v[...,:, :-1] + v[...,:, 1:])/2
    v_f[..., 1:-1:2, 1:-1:2] = (v[...,:-1, :-1] + v[...,1:, :-1] + v[...,:-1, 1:] + v[...,1:, 1:])/4

    return v_f

def interpolateN(v):
    """Interpolate 'v' to the fine grid : Neumann even-sized"""

    nx, ny = v.shape[-2:]
    v_f = torch.zeros(v.shape[:-2] + (2*nx,2*ny), dtype=v.dtype, device=v.device)

    # Interior grid points
    v_f[..., 2:-1:2, 2:-1:2  ] = v[..., 1:, 1:]*9/16 + (v[..., 1:, :-1] + v[..., :-1, 1:]  )*3/16 + v[..., :-1, :-1]*1/16
    v_f[..., 1:-2:2, 2:-1:2  ] = v[..., :-1, 1:]*9/16 + (v[..., :-1, :-1] + v[..., 1:, 1:]  )*3/16 + v[..., 1:, :-1]*1/16
    v_f[..., 2:-1:2, 1:-2:2  ] = v[..., 1:, :-1]*9/16 + (v[..., 1:, 1:] + v[..., :-1, :-1]  )*3/16 + v[..., :-1, 1:]*1/16
    v_f[..., 1:-2:2, 1:-2:2  ] = v[..., :-1, :-1]*9/16 + (v[..., 1:, :-1] + v[..., :-1, 1:]  )*3/16 + v[..., 1:, 1:]*1/16

    # Lateral boundaries
    v_f[..., 0, 2:-1:2  ] = v[..., 0, 1:]*3/4 + v[..., 0, :-1]/4
    v_f[..., 0, 1:-2:2  ] = v[..., 0, :-1]*3/4 + v[..., 0, 1:]/4
    v_f[..., -1, 2:-1:2  ] = v[..., -1, 1:]*3/4 + v[..., -1, :-1]/4
    v_f[..., -1, 1:-2:2  ] = v[..., -1, :-1]*3/4 + v[..., -1, 1:]/4

    v_f[..., 2:-1:2, 0] = v[..., 1: , 0]*3/4 + v[..., :-1, 0]/4
    v_f[..., 1:-2:2, 0] = v[..., :-1 , 0]*3/4 + v[..., 1:, 0]/4
    v_f[..., 2:-1:2, -1] = v[..., 1: , -1]*3/4 + v[..., :-1, -1]/4
    v_f[..., 1:-2:2, -1] = v[..., :-1 , -1]*3/4 + v[..., 1:, -1]/4

    # Corners
    v_f[..., 0, 0] = v[..., 0, 0]
    v_f[..., -1, 0] = v[..., -1, 0]
    v_f[..., 0, -1] = v[..., 0, -1]
    v_f[..., -1, -1] = v[..., -1, -1]

    return v_f

def interpolateN2(v):
    """Interpolate 'v' to the fine grid : Neumann even-sized using ghost points"""

    nx, ny = v.shape[-2:]
    v_f = torch.zeros(v.shape[:-2] + (2*nx,2*ny), dtype=v.dtype, device=v.device)
    v = F.pad(v, (1,1,1,1), mode='replicate')

    # Interior grid points
    v_f[..., :-1:2, :-1:2  ] = v[..., 1:-1, 1:-1]*9/16 + (v[..., 1:-1, :-2] + v[..., :-2, 1:-1]  )*3/16 + v[..., :-2, :-2]*1/16
    v_f[..., 1::2 , :-1:2  ] = v[..., 1:-1, 1:-1]*9/16 + (v[..., 1:-1, :-2] + v[..., 2:, 1:-1]  )*3/16 + v[..., 2:, :-2]*1/16
    v_f[..., :-1:2, 1::2  ]  = v[..., 1:-1, 1:-1]*9/16 + (v[..., 1:-1, 2:] + v[..., :-2, 1:-1]  )*3/16 + v[..., :-2, 2:]*1/16
    v_f[..., 1::2 , 1::2  ]  = v[..., 1:-1, 1:-1]*9/16 + (v[..., 2:, 1:-1] + v[..., 1:-1, 2:]  )*3/16 + v[..., 2:, 2:]*1/16

    return v_f

def choose_restriction(name, boundary_cond):
    restrictions = {
        ('HW', 'dirichlet') : lambda v : restrict(v, mode='HW'),
        ('FW', 'dirichlet') : lambda v : restrict(v, mode='FW'),
        ('four_average', 'neumann') : restrictN
        }

    return restrictions[(name, boundary_cond)]


def choose_interpolation(name, boundary_cond):
    interpolations = {
        ('bilinear', 'dirichlet') : interpolate,
        ('bilinear', 'neumann') : interpolateN2
        }

    return interpolations[(name, boundary_cond)]

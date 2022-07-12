from solver import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__mmain__':
    # DIRICHLET BOUNDARY CONDITIONS
    nlevels    = 8
    NX         = 2*2**(nlevels-1)
    NY         = 2*2**(nlevels-1)

    shape = (NX+1, NY+1)

    device = 'cpu' # 'cuda'
    dtype = torch.float64

    f_orig = torch.zeros(shape, dtype=dtype, device=device)
    u = torch.zeros(shape, dtype=dtype, device=device)
    f = torch.zeros(shape, dtype=dtype, device=device)

    # calculate the RHS and exact solution
    xc = torch.linspace(0, 2*torch.pi,NX+1, dtype=dtype, device=device)
    yc = torch.linspace(0, 2*torch.pi,NY+1, dtype=dtype, device=device)
    DX = xc[1] - xc[0]
    DY = yc[1] - yc[0]
    XX,YY= torch.meshgrid(xc,yc,indexing='ij')

    print('Solving poisson equation with Dirichlet boundary conditions:')
    func = lambda x, y : torch.sin(XX)*torch.sin(YY)
    laplacian = lambda f, dx, dy: (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1])/dx**2 + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1])/dy**2

    sinxsiny = func(XX,YY)
    random = torch.zeros_like(sinxsiny).normal_() * sinxsiny
    f_orig = torch.stack([sinxsiny, random], dim=0)

    delta_f_orig = torch.zeros_like(f_orig)
    delta_f_orig[...,1:-1,1:-1] = laplacian(f_orig, DX, DY)
    # print(f'{NX=}, {NY=}, {nlevels=}')


    mg = MG(dx=DX, dy=DY, nx=NX+1, ny=NY+1, nl=2, nlevels=nlevels, grid_type='vertex-centered', boundary_cond='dirichlet')
    # u, res = mg.solve(delta_f_orig)
    u = torch.zeros_like(delta_f_orig)
    u, res = mg.V_cycle(nlevels, u, delta_f_orig, DX, DY)

    u, f_orig, res = u.cpu(), f_orig.cpu(), res.cpu()
    error = f_orig[...,1:-1, 1:-1] - u[...,1:-1, 1:-1]
    print(f'L_inf (true error), sin(x)sin(y) {np.abs(error[0]).max():.2E}, random {np.abs(error[1]).max():.2E}')

    plt.ion()
    f, a = plt.subplots(1,3)
    f.suptitle('Solving poisson eq. with Full Multi grid, sin(x)sin(y)')
    f.colorbar(a[0].imshow(f_orig[0].cpu()), ax=a[0])
    a[0].set_title('orig. f')
    f.colorbar(a[1].imshow(u[0].cpu()), ax=a[1])
    a[1].set_title('new f')
    f.colorbar(a[2].imshow(np.abs(u-f_orig)[0].cpu()), ax=a[2])
    a[2].set_title('Error')
    plt.tight_layout()

    f, a = plt.subplots(1,3)
    f.suptitle('Solving poisson eq. with Full Multi grid, random func')
    f.colorbar(a[0].imshow(f_orig[1].cpu()), ax=a[0])
    a[0].set_title('orig. f')
    f.colorbar(a[1].imshow(u[1].cpu()), ax=a[1])

    a[1].set_title('new f')
    f.colorbar(a[2].imshow(np.abs(u-f_orig)[1].cpu()), ax=a[2])
    a[2].set_title('Error')
    plt.tight_layout()

if __name__ == '__main__':
    # NEUMANN BOUNDARY CONDITIONS
    #input
    nlevels    = 8
    NX         = 2*2**(nlevels-1)
    NY         = 2*2**(nlevels-1)

    shape = (NX, NY)

    device = 'cpu' # 'cuda'
    dtype = torch.float64

    f_orig = torch.zeros(shape, dtype=dtype, device=device)
    u = torch.zeros(shape, dtype=dtype, device=device)
    f = torch.zeros(shape, dtype=dtype, device=device)

    # calculate the RHS and exact solution
    xc = torch.linspace(0, 2*torch.pi, NX+1, dtype=dtype, device=device)
    yc = torch.linspace(0, 2*torch.pi, NY+1, dtype=dtype, device=device)
    DX = xc[1] - xc[0]
    DY = yc[1] - yc[0]
    xc += DX/2
    yc += DY/2
    XX,YY= torch.meshgrid(xc[:-1],yc[:-1],indexing='ij')

    print('Solving poisson equation:')
    func = lambda x, y : torch.cos(XX)*torch.cos(YY)
    laplacian = lambda f, dx, dy: (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1])/dx**2 + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1])/dy**2

    sinxsiny = func(XX,YY)
    random = torch.zeros_like(sinxsiny).normal_() * sinxsiny
    random -= random.mean()
    f_orig = torch.stack([sinxsiny, random], dim=0)
    delta_f_orig = torch.zeros_like(f_orig)
    delta_f_orig = laplacian(F.pad(f_orig, (1,1,1,1), mode='replicate'), DX, DY)

    # print(f'{NX=}, {NY=}, {nlevels=}')

    # mg = MG(param)
    mg = MG(dx=DX, dy=DY, nx=NX, ny=NY, nl=2, nlevels=nlevels, grid_type='cell-centered', boundary_cond='neumann')
    # mg.smoothing = jacobi_smoothingN2
    u = torch.zeros_like(delta_f_orig, device=device, dtype=dtype)
    u, res = mg.solve(delta_f_orig)
    # u = jacobi_smoothingN2(u, delta_f_orig, DX, DY, 10000)
    # u = torch.zeros_like(delta_f_orig)
    # u, res = mg.V_cycle(nlevels, u, delta_f_orig, DX, DY)

    u, f_orig, res = u.cpu(), f_orig.cpu(), res.cpu()
    error = f_orig[...,1:-1, 1:-1] - u[...,1:-1, 1:-1]
    print(f'L_inf (true error), sin(x)sin(y) {np.abs(error[0]).max():.2E}, random {np.abs(error[1]).max():.2E}')

    plt.ion()
    f, a = plt.subplots(1,3)
    f.suptitle('Solving poisson eq. with Full Multi grid, cos(x)cos(y)')
    f.colorbar(a[0].imshow(f_orig[0].cpu()), ax=a[0])
    a[0].set_title('orig. f')
    f.colorbar(a[1].imshow(u[0].cpu()), ax=a[1])
    a[1].set_title('new f')
    f.colorbar(a[2].imshow(np.abs(u-f_orig)[0].cpu()), ax=a[2])
    a[2].set_title('Error')
    plt.tight_layout()

    f, a = plt.subplots(1,3)
    f.suptitle('Solving poisson eq. with Full Multi grid, random func')
    f.colorbar(a[0].imshow(f_orig[1].cpu()), ax=a[0])
    a[0].set_title('orig. f')
    f.colorbar(a[1].imshow(u[1].cpu()), ax=a[1])

    a[1].set_title('new f')
    f.colorbar(a[2].imshow(np.abs(u-f_orig)[1].cpu()), ax=a[2])
    a[2].set_title('Error')
    plt.tight_layout()

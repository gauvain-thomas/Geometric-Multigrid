import torch
import torch.nn.functional as F

"""Define the smoothing operators for each grid type"""

def jacobi_smoothing(u, f, dx, dy, iters=1, omega=.8, lambd=0.):
    """ω-Jacobi relaxation method for vertex-centered grid with Dirichlet boundary conditions"""

    # Dirichlet BC
    u[..., [0,-1], :] = 0
    u[..., [0,-1]] = 0

    Ax=1.0/dx**2; Ay=1.0/dy**2
    Ap=1.0/(2.0*(Ax+Ay) + lambd)

    for it in range(iters):
        u[...,1:-1,1:-1] = (
                omega*Ap*(Ax*(u[...,2:,1:-1] + u[...,:-2,1:-1])  + Ay*(u[...,1:-1,2:] + u[...,1:-1,:-2]) - f[...,1:-1,1:-1])
                + (1-omega)*u[...,1:-1,1:-1])

        # Dirichlet BC
        u[..., [0,-1], :] = 0
        u[..., [0,-1]] = 0

    return u

def jacobi_smoothingN(u, f, dx, dy, iters=1, omega=.8, lambd=0.):
    """ω-Jacobi relaxation method for cell-centered grid with Neumann boundary conditions without ghost points"""

    Ax=1.0/dx**2; Ay=1.0/dy**2
    Ap  =1.0/(2.0*(Ax+Ay) + lambd)
    Apx =1.0/((Ax+2.0*Ay) + lambd)
    Apy =1.0/((2.0*Ax+Ay) + lambd)
    Apxy=1.0/((Ax+Ay)     + lambd)

    for it in range(iters):
        # Interior grid points
        u[..., 1:-1,1:-1] = (
                omega*Ap*(Ax*(u[...,2:,1:-1] + u[...,:-2,1:-1])  + Ay*(u[...,1:-1,2:] + u[...,1:-1,:-2]) - f[...,1:-1,1:-1])
                + (1-omega)*u[...,1:-1,1:-1])

        # Top boundary
        u[..., 0, 1:-1] = (
                omega*Apx*(Ax*(u[...,1,1:-1])  + Ay*(u[...,0,2:] + u[...,0,:-2]) - f[..., 0, 1:-1])
                + (1-omega)*u[..., 0, 1:-1])

        # Bottom boundary
        u[..., -1, 1:-1] = (
                omega*Apx*(Ax*(u[...,-2,1:-1])  + Ay*(u[...,-1,2:] + u[...,-1,:-2]) - f[..., -1, 1:-1])
                + (1-omega)*u[..., -1, 1:-1])

        # Left boundary
        u[..., 1:-1,0] = (
                omega*Apy*(Ax*(u[...,2:,0] + u[...,:-2,0])  + Ay*(u[...,1:-1,1]) - f[..., 1:-1,0])
                + (1-omega)*u[..., 1:-1,0])

        # Right boundary
        u[..., 1:-1,-1] = (
                omega*Apy*(Ax*(u[...,2:,-1] + u[...,:-2,-1])  + Ay*(u[...,1:-1,-2]) - f[..., 1:-1,-1])
                + (1-omega)*u[..., 1:-1,-1])

        # Corners
        u[..., 0, 0] = (
                omega*Apxy*(Ax*(u[...,1,0])  + Ay*(u[...,0, 1]) - f[...,0,0])
                + (1-omega)*u[...,0,0])
        u[..., -1, 0] = (
                omega*Apxy*(Ax*(u[...,-2,0])  + Ay*(u[...,-1, 1]) - f[...,-1,0])
                + (1-omega)*u[...,-1,0])
        u[..., 0, -1] = (
                omega*Apxy*(Ax*(u[...,1,-1])  + Ay*(u[...,0, -2]) - f[...,0,-1])
                + (1-omega)*u[...,0,-1])
        u[..., -1, -1] = (
                omega*Apxy*(Ax*(u[...,-2,-1])  + Ay*(u[...,-1, -2]) - f[...,-1,-1])
                + (1-omega)*u[...,-1,-1])

        u -= torch.mean(u, dim=[-1,-2], keepdim=True)

    return u

def jacobi_smoothingN2(u, f, dx, dy, iters=1, omega=.8, lambd=0.):
    """ω-Jacobi relaxation method for cell-centered grid with Neumann boundary conditions using ghost points"""

    Ax=1.0/dx**2; Ay=1.0/dy**2
    Ap  =1.0/(2.0*(Ax+Ay) + lambd)

    f = F.pad(f, (1,1,1,1))

    for it in range(iters):
        u = F.pad(u, (1,1,1,1), mode='replicate')
        # Interior grid points
        u[..., 1:-1,1:-1] = (
                omega*Ap*(Ax*(u[...,2:,1:-1] + u[...,:-2,1:-1])  + Ay*(u[...,1:-1,2:] + u[...,1:-1,:-2]) - f[...,1:-1,1:-1])
                + (1-omega)*u[...,1:-1,1:-1])

        u = u[..., 1:-1, 1:-1]
        u -= torch.mean(u, dim=[-1,-2], keepdim=True)

    return u


def GS_RB_smoothing(u, f, dx, dy, iters=1, omega=1.22, lambd=0.):
    """Gauss-Seidel red-black relaxation method for vertex-centered grid with Dirichlet boundary conditions"""

    #Dirichlet BC
    u[..., [0,-1], :] = 0
    u[..., [0,-1]]    = 0

    Ax=1.0/dx**2; Ay=1.0/dy**2
    Ap=1.0/(2.0*(Ax+Ay) + lambd)

    for it in range(iters):
        u[..., 2:-2:2, 1::2] = omega*Ap*(Ax*(u[..., 1:-3:2, 1::2] + u[..., 3:-1:2, 1::2])
                                    + Ay*(u[..., 2:-2:2, :-1:2] + u[..., 2:-2:2, 2::2])
                                    - f[..., 2:-2:2, 1::2]) + (1-omega)*u[..., 2:-2:2, 1::2]

        u[..., 1::2, 2:-2:2] = omega*Ap*(Ax*(u[..., :-1:2, 2:-2:2] + u[..., 2::2, 2:-2:2])
                                    + Ay*(u[..., 1::2, 3:-1:2] + u[..., 1::2, 1:-3:2])
                                    - f[..., 1::2, 2:-2:2]) + (1-omega)*u[..., 1::2, 2:-2:2]



        u[..., 1:-1:2, 1::2] = omega*Ap*(Ax*(u[..., :-1:2, 1::2] + u[..., 2::2, 1::2])
                                    + Ay*(u[..., 1:-1:2, :-1:2] + u[..., 1:-1:2, 2::2])
                                    - f[..., 1:-1:2, 1::2]) + (1-omega)*u[..., 1:-1:2, 1::2]

        u[..., 2:-1:2, 2:-1:2] = omega*Ap*(Ax*(u[..., 1:-2:2, 2:-1:2] + u[..., 3::2, 2:-1:2])
                                    + Ay*(u[..., 2:-1:2, 1:-2:2] + u[..., 2:-1:2, 3::2])
                                    - f[..., 2:-1:2, 2:-1:2]) + (1-omega)*u[..., 2:-1:2, 2:-1:2]

        #Dirichlet BC
        u[..., [0,-1], :] = 0
        u[..., [0,-1]]    = 0

    return u

def GS_RB_smoothingN(u, f, dx, dy, iters=1, omega=1.22, lambd=0.):
    """Gauss-Seidel red-black relaxation method for cell-centered grid with Neumann boundary conditions without ghost points"""

    Ax  = 1.0/dx**2; Ay=1.0/dy**2
    Ap  = 1.0/(2.0*(Ax+Ay) + lambd)
    Apx = 1.0/((Ax+2.0*Ay) + lambd)
    Apy = 1.0/((2.0*Ax+Ay) + lambd)
    Apxy= 1.0/((Ax+Ay)     + lambd)

    for it in range(iters):
        # Black : Interior / Lateral / Corners

        u[..., 2:-1:2, 1:-2:2] = omega*Ap*(Ax*(u[..., 1:-2:2, 1:-2:2] + u[..., 3::2, 1:-2:2])
                                    + Ay*(u[..., 2:-1:2, :-3:2] + u[..., 2:-1:2, 2:-1:2])
                                    - f[..., 2:-1:2, 1:-2:2]) + (1-omega)*u[..., 2:-1:2, 1:-2:2]

        u[..., 1:-2:2, 2:-1:2] = omega*Ap*(Ax*(u[..., :-3:2, 2:-1:2] + u[..., 2:-1:2, 2:-1:2])
                                    + Ay*(u[..., 1:-2:2, 3::2] + u[..., 1:-2:2, 1:-2:2])
                                    - f[..., 1:-2:2, 2:-1:2]) + (1-omega)*u[..., 1:-2:2, 2:-1:2]
        # Up
        u[..., 0, 1:-2:2] = omega*Apx*(Ax*(u[..., 1, 1:-2:2])
                                    + Ay*(u[..., 0, :-3:2] + u[..., 0, 2:-1:2])
                                    - f[..., 0, 1:-2:2]) + (1-omega)*u[..., 0, 1:-2:2]
        # Right
        u[..., 2:-1:2, -1] = omega*Apy*(Ax*(u[..., 1:-2:2, -1] + u[..., 3::2, -1])
                                    + Ay*(u[..., 2:-1:2, -2])
                                    - f[..., 2:-1:2, -1]) + (1-omega)*u[..., 2:-1:2, -1]
        # Bottom
        u[..., -1, 2:-1:2] = omega*Apx*(Ax*(u[..., -2, 2:-1:2])
                                    + Ay*(u[..., -1, 3::2] + u[..., -1, 1:-2:2])
                                    - f[..., -1, 2:-1:2]) + (1-omega)*u[..., -1, 2:-1:2]
        # Left
        u[..., 1:-2:2, 0] = omega*Apy*(Ax*(u[..., :-3:2, 0] + u[..., 2:-1:2, 0])
                                    + Ay*(u[..., 1:-2:2, 1])
                                    - f[..., 1:-2:2, 0]) + (1-omega)*u[..., 1:-2:2, 0]
        # Top right
        u[..., 0, -1] = omega*Apxy*(Ax*(u[..., 1, -1])
                                    + Ay*(u[..., 0, -2])
                                    - f[..., 0, -1]) + (1-omega)*u[..., 0, -1]
        # Bottom left
        u[..., -1, 0] = omega*Apxy*(Ax*(u[..., -2, 0])
                                    + Ay*(u[..., -1, 1])
                                    - f[..., -1, 0]) + (1-omega)*u[..., -1, 0]



        # Red : Interior / Lateral / Corners

        u[..., 1:-2:2, 1:-2:2] = omega*Ap*(Ax*(u[..., :-3:2, 1:-2:2] + u[..., 2:-1:2, 1:-2:2])
                                    + Ay*(u[..., 1:-2:2, :-3:2] + u[..., 1:-2:2, 2:-1:2])
                                    - f[..., 1:-2:2, 1:-2:2]) + (1-omega)*u[..., 1:-2:2, 1:-2:2]

        u[..., 2:-1:2, 2:-1:2] = omega*Ap*(Ax*(u[..., 1:-2:2, 2:-1:2] + u[..., 3::2, 2:-1:2])
                                    + Ay*(u[..., 2:-1:2, 1:-2:2] + u[..., 2:-1:2, 3::2])
                                    - f[..., 2:-1:2, 2:-1:2]) + (1-omega)*u[..., 2:-1:2, 2:-1:2]
        # Right
        u[..., 1:-2:2, -1] = omega*Apy*(Ax*(u[..., :-2:2, -1] + u[..., 2::2, -1])
                                    + Ay*(u[..., 1:-1:2, -2])
                                    - f[..., 1:-1:2, -1]) + (1-omega)*u[..., 1:-1:2, -1]
        # Bottom
        u[..., -1, 1:-2:2] = omega*Apx*(Ax*(u[..., -2, 1:-2:2])
                                    + Ay*(u[..., -1, :-3:2] + u[..., -1, 2:-1:2])
                                    - f[..., -1, 1:-2:2]) + (1-omega)*u[..., -1, 1:-2:2]
        # Up
        u[..., 0, 2:-1:2] = omega*Apx*(Ax*(u[..., 1, 2:-1:2])
                                    + Ay*(u[..., 0, 1:-2:2] + u[..., 0, 3::2])
                                    - f[..., 0, 2:-1:2]) + (1-omega)*u[..., 0, 2:-1:2]
        # Left
        u[..., 2:-1:2, 0] = omega*Apy*(Ax*(u[..., 1:-2:2, 0] + u[..., 3::2, 0])
                                    + Ay*(u[..., 2:-1:2, 1])
                                    - f[..., 2:-1:2, 0]) + (1-omega)*u[..., 2:-1:2, 0]
        # Top left
        u[..., 0, 0] = omega*Apxy*(Ax*(u[..., 1, 0])
                                    + Ay*(u[..., 0, 1])
                                    - f[..., 0, 0]) + (1-omega)*u[..., 0, 0]
        # Bottom right
        u[..., -1, -1] = omega*Apxy*(Ax*(u[..., -2, -1])
                                    + Ay*(u[..., -1, -2])
                                    - f[..., -1, -1]) + (1-omega)*u[..., -1, -1]


        u -= torch.mean(u, dim=[-1,-2], keepdim=True)

    return u

def GS_RB_smoothingN2(u, f, dx, dy, iters=1, omega=1.22, lambd=0.):
    """Gauss-Seidel red-black relaxation method for cell-centered grid with Neumann boundary conditions using ghost points"""

    Ax  = 1.0/dx**2; Ay=1.0/dy**2
    Ap  = 1.0/(2.0*(Ax+Ay) + lambd)

    f = F.pad(f, (1,1,1,1))

    for it in range(iters):
        u = F.pad(u, (1,1,1,1), mode='replicate')

        u[..., 2:-1:2, 1:-2:2] = omega*Ap*(Ax*(u[..., 1:-2:2, 1:-2:2] + u[..., 3::2, 1:-2:2])
                                    + Ay*(u[..., 2:-1:2, :-3:2] + u[..., 2:-1:2, 2:-1:2])
                                    - f[..., 2:-1:2, 1:-2:2]) + (1-omega)*u[..., 2:-1:2, 1:-2:2]

        u[..., 1:-2:2, 2:-1:2] = omega*Ap*(Ax*(u[..., :-3:2, 2:-1:2] + u[..., 2:-1:2, 2:-1:2])
                                    + Ay*(u[..., 1:-2:2, 3::2] + u[..., 1:-2:2, 1:-2:2])
                                    - f[..., 1:-2:2, 2:-1:2]) + (1-omega)*u[..., 1:-2:2, 2:-1:2]

        u[..., 1:-2:2, 1:-2:2] = omega*Ap*(Ax*(u[..., :-3:2, 1:-2:2] + u[..., 2:-1:2, 1:-2:2])
                                    + Ay*(u[..., 1:-2:2, :-3:2] + u[..., 1:-2:2, 2:-1:2])
                                    - f[..., 1:-2:2, 1:-2:2]) + (1-omega)*u[..., 1:-2:2, 1:-2:2]

        u[..., 2:-1:2, 2:-1:2] = omega*Ap*(Ax*(u[..., 1:-2:2, 2:-1:2] + u[..., 3::2, 2:-1:2])
                                    + Ay*(u[..., 2:-1:2, 1:-2:2] + u[..., 2:-1:2, 3::2])
                                            - f[..., 2:-1:2, 2:-1:2]) + (1-omega)*u[..., 2:-1:2, 2:-1:2]

        u = u[..., 1:-1, 1:-1]

        u -= torch.mean(u, dim=[-1,-2], keepdim=True)

    return u


def residual(u, f, dx, dy, lambd=0.):
    """ Compute the residual for Dirichlet boundary counditions """
    Ax = 1.0/dx**2
    Ay = 1.0/dy**2

    res = f[...,1:-1,1:-1] - (( Ax*(u[...,2:,1:-1] + u[...,:-2,1:-1])
                                   + Ay*(u[...,1:-1,2:] + u[...,1:-1,:-2])
                                   - (2.0*(Ax+Ay) + lambd)*u[...,1:-1,1:-1]))
    return F.pad(res, (1,1,1,1))

def residualN(u, f, dx, dy, lambd=0.):
    """ Compute the residual for Neumann boundary counditions """
    Ax = 1.0/dx**2
    Ay = 1.0/dy**2
    u = F.pad(u, (1,1,1,1), mode='replicate')

    res = f - (( Ax*(u[...,2:,1:-1] + u[...,:-2,1:-1])
                                   + Ay*(u[...,1:-1,2:] + u[...,1:-1,:-2])
                                   - (2.0*(Ax+Ay) + lambd)*u[...,1:-1,1:-1]))

    return res

def choose_smoother(name, boundary_cond):
    smoothers = {
        ('jacobi', 'dirichlet') : jacobi_smoothing,
        ('jacobi', 'neumann') : jacobi_smoothingN2,
        ('gauss_seidel_rb', 'dirichlet') : GS_RB_smoothing,
        ('gauss_seidel_rb', 'neumann') : GS_RB_smoothingN2
    }

    return smoothers[(name, boundary_cond)]

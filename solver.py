"""
A python-numpy implementation for a geometric multigrid Poisson/Helmoltz solver
Equations are of the form : Δu - λu = f
"""

from smoothers import choose_smoother, residual, residualN
from scaling import choose_restriction, choose_interpolation
import torch

class MG():
    """Multrigrid solver for Laplace / Helmoltz like equations :
    Δu - λu = f with homogeneous Dirichlet/Neumann boundary conditions
    Currently supported grid types and boundary conditions combinations are :
        - Dirichlet : the grid is vertex-centered and must be odd-sized.
        - Neumann : the grid is cell-centered and must be even-sized.
    """

    def __init__(self, *, dx, dy, nx, ny, nl, nlevels, grid_type, boundary_cond,
                 lambd=0.,
                 tol=1e-4,
                 max_ite=15,
                 presmoother=('gauss_seidel_rb', {'omega':1.2, 'iters':2}),
                 postsmoother=('gauss_seidel_rb', {'omega':1.2, 'iters':2}),
                 restriction=None,
                 interpolation='bilinear',
                 dtype=torch.float64,
                 device='cpu',
                 compiled=True):
        """
        Parameters :
        ------------
        dx, dy : float
            Initial mesh size
        nx, ny : int
            Inital grid size
        nl : int
            Number of layers (stacked upon first dimension of tensor)
        nlevels :
            Depth of multigrid cycle (i.e. number of restrictions/interpolations)
        grid_type : {'vertex-centered', 'cell-centered'}
        lambd : float or torch.Tensor
            λ parameter in Helmholtz equation. λ=0 for Poisson equations.
            In case of a multi-layer grid, λ can be a tensor of each λ for each grid.
        tol : float
            Tolerance for stopping criteria in FMG cycle
        max_ite : int
            Maximum number of V-cycles to perform at the end of FMG cycle if
            tolerance is not reached
        presmoother : (string, dict)
            Describe the presmoother used in first half of the multigrid cycle.
            Must be of the form : ('smoother_name', {'omega':omega, 'iters':iters})
            Currently only 'jacobi' and 'gauss_seidel_rb' are supported.
        postsmoother : (string, dict)
            Same as for presmoother, used in the second half of the multigrid cycle.
        restriction : {'HW', 'FW', 'bilinear'}
            Transfer operator used for restriction.
        interpolation : {'bilinear'}
            Transfer operator used for restriction.
        dtype : torch.dtype
            Type used for torch tensors
        device : {'cpu', 'cuda'}
            Device on which are performed calculations
        compiled : bool
            Choose whether to compile using torch.jit.trace or not.
        """

        self.dtype = dtype
        self.device = device

        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.nl = nl
        self.dx = torch.full((), dx, dtype=self.dtype, device=self.device)
        self.dy = torch.full((), dy, dtype=self.dtype, device=self.device)

        self.lambd = lambd if isinstance(lambd, torch.Tensor) else torch.full((), lambd, dtype=self.dtype, device=self.device)

        assert boundary_cond in ['dirichlet', 'neumann'], "Boundary condition must be 'dirichlet' or 'neumann'"

        self.nlevels = nlevels
        self.max_ite = max_ite
        self.tol = tol

        self.ns = 15 # Number of bottom smoothing (exact solution)

        self.shape = (self.nl, self.nx, self.ny) # if self.nlevels > 1 else (self.nx, self.ny)
        if boundary_cond=='dirichlet' :
            assert self.nx%2**self.nlevels==1 and self.ny%2**self.nlevels==1, f"Grid size {(self.nx, self.ny)} incorrect"
        if boundary_cond=='neumann':
            assert self.nx%2**self.nlevels==0 and self.ny%2**self.nlevels==0, f"Grid size {(self.nx, self.ny)} incorrect"

        print(f'PyTorch multigrid solver, {self.device}, {self.dtype}')

        v = torch.zeros(self.shape, dtype=self.dtype, device=self.device)

        # Setup smoothers
        smoother, param = presmoother
        presmoother = choose_smoother(smoother, boundary_cond)
        self.presmooth = lambda u, f, dx, dy : presmoother(u, f, dx, dy,
                             lambd=self.lambd, omega=param['omega'], iters=param['iters'])
        if compiled : self.presmooth = torch.jit.trace(self.presmooth, (v, v, self.dx, self.dy))

        smoother, param = postsmoother
        postsmoother = choose_smoother(smoother, boundary_cond)
        self.postsmooth = lambda u, f, dx, dy : postsmoother(u, f, dx, dy,
                             lambd=self.lambd, omega=param['omega'], iters=param['iters'])
        if compiled : self.postsmooth = torch.jit.trace(self.postsmooth, (v, v, self.dx, self.dy))

        # Setup transfer operators
        default_restriction = {'dirichlet' : 'FW', 'neumann' : 'four_average'}
        if not restriction : restriction = default_restriction[boundary_cond]
        self.restrict = choose_restriction(restriction, boundary_cond)
        self.interpolate = choose_interpolation(interpolation, boundary_cond)
        if compiled : self.restrict    = torch.jit.trace(self.restrict, (v,))
        if compiled : self.interpolate = torch.jit.trace(self.interpolate, (v,))

        # Setup residual
        res = residual if boundary_cond=='dirichlet' else residualN
        self.residual = lambda u, f, dx, dy : res(u, f, dx, dy, self.lambd)
        if compiled : self.residual = torch.jit.trace(self.residual, (v, v, self.dx, self.dy))

        print('Initialization completed')

    def solve(self, f):
        return self.FMG(f, self.dx, self.dy)

    def _solveV(self, f):
        u = torch.zeros_like(f, dtype=self.dtype, device=self.device)
        u, res = self.V_cycle(self.nlevels, u, f, self.dx, self.dy)
        nres = res.norm()/f.norm()
        while nres > self.tol:
            u, res = self.V_cycle(self.nlevels, u, f, self.dx, self.dy)
            nres = res.norm()/f.norm()
        return u, res

    def Two_cycles(self, u, f, dx, dy):

        #Step 1: Relax Au=f on this grid
        u = self.presmooth(u ,f, dx, dy)
        res = self.residual(u, f, dx, dy)

        #Step 2: Restrict residual to coarse grid
        res_c = self.restrict(res)

        #Step 3:Solve A e_c=res_c on the coarse grid
        e_c = torch.zeros_like(res_c)
        e_c = self.presmooth(e_c, res_c, dx*2, dy*2)

        #Step 4: Interpolate(prolong) e_c to fine grid and add to u
        u += self.interpolate(e_c)

        #Step 5: Relax Au=f on this grid
        u = self.postsmooth(u, f, dx, dy)
        res = self.residual(u, f, dx, dy)
        return u, res


    def V_cycle(self, num_levels, u, f, dx, dy, level=1):

        if(level==num_levels): #bottom solve
            for _ in range(self.ns):
                u = self.presmooth(u, f, dx, dy)
            res = self.residual(u, f, dx, dy)
            return u,res

        # Step 1: Relax Au=f on this grid
        u = self.presmooth(u, f, dx, dy)
        res = self.residual(u, f, dx, dy)

        # Step 2: Restrict residual to coarse grid
        res_c = self.restrict(res)

        # Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
        e_c = torch.zeros_like(res_c)
        e_c, res_c = self.V_cycle(num_levels, e_c, res_c, dx*2, dy*2, level=level+1)

        # Step 4: Interpolate(prolong) e_c to fine grid and add to u
        u += self.interpolate(e_c)

        # Step 5: Relax Au=f on this grid
        u = self.postsmooth(u, f, dx, dy)
        res = self.residual(u, f, dx, dy)

        return u, res

    def FMG(self, f, dx, dy, level=1):
        """Full Multigrid cycle"""

        if(level==self.nlevels):#bottom solve
            u = torch.zeros_like(f)
            for _ in range(self.ns):
                u = self.presmooth(u, f, dx, dy)
            res = self.residual(u, f, dx, dy)
            return u, res

        # Step 1: Restrict the rhs to a coarse grid
        f_c = self.restrict(f)

        # Step 2: Solve the coarse grid problem using FMG
        u_c, _ = self.FMG(f_c, dx*2, dy*2, level+1)

        # Step 3: Interpolate u_c to the fine grid
        u = self.interpolate(u_c)

        # Step 4: Execute 'nv' V-cycles
        # for _ in range(self.nv):
        #     u, res = self.V_cycle(self.nlevels-level, u, f, dx, dy)

        u, res = self.V_cycle(self.nlevels-level, u, f, dx, dy)

        if level < 2:
            nres = res.norm()/f.norm()
            nite = 0
            while nite < self.max_ite and nres > self.tol:
                u, res = self.V_cycle(self.nlevels-level, u, f, dx, dy)
                nres = res.norm()/f.norm()
                nite += 1

        return u, res

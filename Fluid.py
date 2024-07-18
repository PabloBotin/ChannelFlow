import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Fluid:

    def __init__(self, numY, ratio):
        self.ratio= ratio
        self.numX = numY*ratio
        self.numY = numY
        self.H=1
        self.L= ratio

        # Initialize arrays. 
        self.u = np.zeros((self.numY, self.numX))  # First, fill the entire array with zeros 
        self.v = np.zeros((self.numY, self.numX))  # First, fill the entire array with zeros
        self.p = np.ones((self.numY, self.numX))  # Pressure field
        # Initialize arrays. 
        # Set Boundaries.  1:Fluid/2:solid.
        self.s = np.ones((self.numY, self.numX))  # All fluid. 
        #self.s[0, :] = 0  # Bottom wall
        #self.s[-1, :] = 0  # Top wall

        # Calculate body force distribution
        F= -120
        self.AI = np.zeros_like(self.u) # Init force array. 
        mid_y, mid_x = self.numY // 2, self.numX // 2 # 20 , 20. In the middle of the domain. 
        num_cells = 3  # Number of cells in y-direction (total 6 cells: 3 before and 3 after the midpoint)
        # Assign force to corresponding cells. 
        self.AI[(mid_y - num_cells):(mid_y + num_cells),mid_x] = F

        # Define steps 
        self.dx = 2 / (self.numX - 1)
        self.dy = 2 / (self.numY - 1)
        self.nu= 0.1
        #self.dt= .00002 # I made it smaller due to stability issues. 
        self.dt= self.compute_time_step(self.u, self.v, self.dx, self.dy, self.nu, cfl_number=0.7)
 
    def Momentum (self, p):
        # Set parameters
        nu= self.nu
        rho=1
        # Initialize arrays 
        u, v, dt, dx, dy, s= self.u, self.v, self.dt, self.dx, self.dy, self.s
        un = u.copy()
        vn = v.copy()

        # The loop excludes the boundaries. 
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]- 
                        s[1:-1, 1:-1]*un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         s[1:-1, 1:-1]*vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         s[1:-1, 1:-1]*dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         s[1:-1, 1:-1]* nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] - 
                        s[1:-1, 1:-1]*un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        s[1:-1, 1:-1]*vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        s[1:-1, 1:-1]*dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        s[1:-1, 1:-1]*nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # BC's. Non-Slip (Dirichlet)
        # Inlet, Dirichlet BC.
        u[1:-1, 0] = 2
        v[1:-1, 0] = 0
        # Outlet, Neumann. du/dx=0.
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        # Bottom 
        u[0, :]  = 0
        v[0, :]  = 0
        # Top
        u[-1, :] = 0    
        v[-1, :] = 0
    
    def build_b(self, rho, dt, u, v, dx, dy): 
            b = np.zeros_like(self.u)
            b[1:-1, 1:-1] = (rho * (1 / dt * 
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                            (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
            #print ("b: ", np.sum(b))
            return b
    
    def pressure_poisson(self, p):
        u, v, dx, dy, dt = self.u, self.v, self.dx, self.dy, self.dt
        rho=1 
        F=-2
        # Build b term. 
        b= self.build_b(rho, dt, u, v, dx, dy)

        # Initialize temporary arrays. 
        for i in range(5): # Adjust number of iterations. 
            pn = p.copy()
            # Check this: 
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

            # InletNeumann BCs: dp/dx = 0 at x = 0
            p[:, 0] = p[:, 1]
            #p[:, 0] = 2 # For pouiseuille comparison. 
            # Outlet, Dirichlet BC: p = 2 at x = 2
            p[:, -1] = 2
            #p[:, -1] = 0 # For Poiseuille comparison.  
            # Top and Bottom. Neumann dp/dy=0 
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[-1, :] = p[-2, :]   # dp/dy = 0 at y = yMax

            
        return p 

    def CFL_Check(self): 
        cfl= self.dt / self.dx * max(np.max(np.abs(self.u)), np.max(np.abs(self.v)))
        if cfl > 1.0:
            raise ValueError("CFL condition violated. Consider reducing the time step size self.dt.")

    def Div(self):
        # Compute the partial derivatives
        dudx = np.gradient(self.u, axis=1) / self.dx
        dvdy = np.gradient(self.v, axis=0) / self.dy
        # Compute the divergence
        div = dudx + dvdy
        max_div = np.max(np.abs(div))
        sum_div = np.sum(np.abs(div))
        print ("Max divergence: ", max_div)
        print ("Sum divergence: ", sum_div)

    def AddSquare(self, L):
        # Calculate the center start indices for symmetric placement
        i = int((self.numX * 3/7) - (L / 2))
        j = (self.numY - L) // 2

        # Set the specified square area to solid (s = 0)
        self.s[j:j + L, i:i + L] = 0

    def compute_time_step(self, u, v, dx, dy, nu, cfl_number=0.1):
        """
        Compute the maximum allowable time step for a 2D incompressible Navier-Stokes simulation.
        I could use this function to automatically adjust the timestep based on the current velocity field. 
        
        Parameters:
            nu (float): Kinematic viscosity.
            cfl_number (float): CFL number (typically < 1 for stability).
            
        Returns:
            float: Maximum allowable time step.
        """
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        
        dt_conv_x = dx / u_max if u_max != 0 else np.inf
        dt_conv_y = dy / v_max if v_max != 0 else np.inf
        dt_diff_x = dx**2 / (2 * nu)
        dt_diff_y = dy**2 / (2 * nu)
        
        dt = cfl_number * min(dt_conv_x, dt_conv_y, dt_diff_x, dt_diff_y)
        
        return dt
        #self.dt= dt
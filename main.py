from Fluid import Fluid
from Visualize import Visualize
import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters to set. 
ratio = 2 # L = ratio*H. 
numY = 40 # Mesh resolution. Sets the number of vertical cells. All cells are square. 
SquareWidth = 0 # Set the width of the square or set to zero for no square.
# Need to review the way the presence of the square affects the flow.  

def main():

    sim = Fluid(numY, ratio) # Create instance of the class.
    sim.CFL_Check() # Check CFL condition 
    if SquareWidth!= 0:
        sim.AddSquare(SquareWidth)

    # Solve:
    p= sim.p
    for t in range(2000):
        p = sim.pressure_poisson(p)
        sim.Momentum(p)
        # Update timestep:
        sim.dt= sim.compute_time_step(sim.u, sim.v, sim.dx, sim.dy, sim.nu, cfl_number=0.2)

    # Print max and sum divergence. 
    #sim.Div()

    # Visualize
    #Visualize.plot_Quiver(sim.p, sim.u, sim.v, sim.numX, sim.numY, sim.s)
    Visualize.plot_Vel(sim.u, sim.v, sim.numX, sim.numY)
    Visualize.plot_divergence(sim.u, sim.v, sim.numX, sim.numY)

    # Comparison with Poiseuille flow analytical solution. Need to set a dp/dx=cte and long pipe! 
    #Visualize.PoiseuilleValidation (sim.u[:, -1], sim.numY, sim.L, sim.H)

if __name__ == "__main__": # Only run the functions defined in this code. 
    main()

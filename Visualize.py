import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualize:
    def __init__(self, fluid_instance):
        self.fluid_instance = fluid_instance

    def plot_Turbine(p, u, v, nx, ny, AI):
        L=1
        Ratio=nx/ny
        x = np.linspace(0, Ratio*L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        plot_type = "quiver"  # "quiver" or "streamplot"
        
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        
        # Plotting the pressure field as a contour
        contourf = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        fig.colorbar(contourf, ax=ax)
        
        # Plotting the pressure field outlines
        ax.contour(X, Y, p, cmap=cm.viridis)
        
        # Plotting velocity field
        if plot_type == "quiver":
            ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
        elif plot_type == "streamplot":
            ax.streamplot(X, Y, u, v)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Mask and overlay the region with non-zero AI values
        masked_AI = np.ma.masked_where(AI == 0, AI)
        black_cmap = ListedColormap(['black'])
        
        ax.imshow(masked_AI, cmap=black_cmap, origin='lower', alpha=1, extent=[0, 2, 0, 2])
        
        plt.show()

    def plot_Vel(u, v, nx, ny):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        
        # Compute the velocity magnitude
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Plotting the velocity field as colored cells
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        c = ax.pcolormesh(X, Y, velocity_magnitude, shading='auto', cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(c, cax=cax).set_label('Velocity Magnitude')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Velocity Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_divergence(u, v, nx, ny):
        """
        Plots the 2D divergence field given the velocity components u and v.
        """
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        dx = L / (nx - 1)
        dy = L / (ny - 1)
        
        # Compute the partial derivatives
        dudx = np.gradient(u, axis=1) / dx
        dvdy = np.gradient(v, axis=0) / dy
        
        # Compute the divergence
        divergence = dudx + dvdy
        
        # Plot the divergence field
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        cp = ax.contourf(X, Y, divergence, 20, cmap='viridis')
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(cp, cax=cax).set_label('Divergence')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D Divergence Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def PoiseuilleValidation (u_numerical, ny, L, H):
        def compute_analytical_velocity(y, h, mu, dpdx):
            return (1 / (2 * mu)) * dpdx * (h**2 - y**2)

        def plot_velocity_profiles(y, u_analytical, u_numerical):
            plt.figure(figsize=(10, 6))
            plt.plot(u_analytical, y, label='Analytical Solution', linewidth=5)
            plt.plot(u_numerical, y, label='Numerical Solution at x=L', linewidth=4)
            plt.xlabel('Velocity $u$')
            plt.ylabel('Position $y$')
            plt.title('Comparison of Analytical (Poiseuille) and Numerical Velocity Profiles')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Parameters
        h2 = H/2  # Half-distance between the plates
        mu = 0.01  # Dynamic viscosity
        L = L  # Length of the channel
        y = np.linspace(-h2, h2, ny)  # y-coordinates
        # Overall pressure drop from your simulation
        p_in = 2.0  # Example inlet pressure
        p_out = 0.0  # Example outlet pressure
        dp = p_in - p_out
        # Effective pressure gradient
        dpdx = dp / L
        # Compute analytical solution
        u_analytical = compute_analytical_velocity(y, h2, mu, dpdx)
        # Plotting the velocity profiles
        plot_velocity_profiles(y, u_analytical, u_numerical)

    def plot_Vel(u, v, nx, ny):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        
        # Compute the velocity magnitude
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Plotting the velocity field as colored cells
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        c = ax.pcolormesh(X, Y, velocity_magnitude, shading='auto', cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(c, cax=cax).set_label('Velocity Magnitude')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Velocity Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_Quiver(p, u, v, nx, ny, s, quiver_density=20, quiver_width=0.002):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        plot_type = "quiver"  # "quiver" or "streamplot"
        
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        
        # Plotting the pressure field as a contour
        contourf = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the colorbar
        fig.colorbar(contourf, cax=cax)
        
        # Plotting the pressure field outlines
        ax.contour(X, Y, p, cmap=cm.viridis)
        
        # Plotting velocity field
        step = max(1, nx // quiver_density)
        if plot_type == "quiver":
            ax.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step], v[::step, ::step], width=quiver_width) 
        elif plot_type == "streamplot":
            ax.streamplot(X, Y, u, v)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Mask and overlay the region
        masked_s = np.ma.masked_where(s != 0, s)
        black_cmap = ListedColormap(['orange'])
        
        # Adjust extent to match the domain
        extent = [0, Ratio * L, 0, L]
        ax.imshow(masked_s, cmap=black_cmap, origin='lower', alpha=1, extent=extent)
        
        # Ensure the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()
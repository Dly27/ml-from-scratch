import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import laplace


class ThermalSource:
    def __init__(self,
                 source_type="none",
                 A=50,
                 center=(0.5, 0.5),
                 sigma=0.02,
                 R=0.3,
                 omega=2 * np.pi / 50,
                 t_pulse=20):
        self.source_type = source_type
        self.A = A
        self.center = center
        self.sigma = sigma
        self.R = R
        self.omega = omega
        self.t_pulse = t_pulse

    def stationary_gaussian(self, X, Y):
        x0, y0 = self.center
        return self.A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * self.sigma**2))

    def lateral_gaussian(self, X, Y, t):
        x0, y0 = self.center
        x_t = x0 + self.R * np.sin(self.omega * t)
        return self.A * np.exp(-((X - x_t)**2 + (Y - y0)**2) / (2 * self.sigma**2))

    def circular_gaussian(self, X, Y, t):
        x0, y0 = self.center
        x_t = x0 + self.R * np.sin(self.omega * t)
        y_t = y0 + self.R * np.cos(self.omega * t)
        return self.A * np.exp(-((X - x_t)**2 + (Y - y_t)**2) / (2 * self.sigma**2))

    def uniform(self, X, Y):
        return self.A * np.ones_like(X)

    def pulsed(self, X, Y, t):
        if (t % self.t_pulse) < self.t_pulse // 2:
            x0, y0 = self.center
            return self.A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * self.sigma**2))
        else:
            return np.zeros_like(X)

    def get_value(self, X, Y, t):
        method_map = {
            "stationary_gaussian": lambda: self.stationary_gaussian(X, Y),
            "lateral_gaussian": lambda: self.lateral_gaussian(X, Y, t),
            "circular_gaussian": lambda: self.circular_gaussian(X, Y, t),
            "uniform": lambda: self.uniform(X, Y),
            "pulsed": lambda: self.pulsed(X, Y, t),
        }
        return method_map.get(self.source_type, lambda: np.zeros_like(X))()

class Grid:
    def __init__(self, width, height, step, dt, nt):
        self.width = width
        self.height = height
        self.dx = step
        self.dy = step
        self.dt = dt
        self.nt = nt
        self.temp_matrix = None
        self.nx = int(width / step)
        self.ny = int(height / step)
        self.temp_matrix = np.zeros((self.nx, self.ny))
        x = np.linspace(0, self.width, self.nx)
        y = np.linspace(0, self.height, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.sources = []
        self.alpha_matrix = None
        self.k_matrix = None
        self.rho_matrix = None
        self.c_matrix = None


    def set_boundaries(self, left, right, top, bottom, boundary_type):
        if boundary_type == "dirichlet":
            self.temp_matrix[0, :] = left
            self.temp_matrix[-1, :] = right
            self.temp_matrix[:, 0] = bottom
            self.temp_matrix[:, -1] = top

        elif boundary_type == "zero_gradient_neumann":
            self.temp_matrix[0, :] = self.temp_matrix[1, :]
            self.temp_matrix[-1, :] = self.temp_matrix[-2, :]
            self.temp_matrix[:, 0] = self.temp_matrix[:, 1]
            self.temp_matrix[:, -1] = self.temp_matrix[:, -2]

    def create_alpha_matrix(self, k, rho, c):
        self.k_matrix = np.ones_like(self.temp_matrix) * k
        self.rho_matrix = np.ones_like(self.temp_matrix) * rho
        self.c_matrix = np.ones_like(self.temp_matrix) * c
        self.alpha_matrix = self.k_matrix / (self.rho_matrix * self.c_matrix)

    def set_material_region(self, region_mask_fn, k=None, rho=None, c=None):
        mask = region_mask_fn(self.X, self.Y)
        if k is not None:
            self.k_matrix[mask] = k
        if rho is not None:
            self.rho_matrix[mask] = rho
        if c is not None:
            self.c_matrix[mask] = c
        self.alpha_matrix = self.k_matrix / (self.rho_matrix * self.c_matrix)

    def add_source(self, source):
        self.sources.append(source)

    def update(self, t):
        temp_matrix_new = self.temp_matrix.copy()
        laplacian = laplace(self.temp_matrix, mode="nearest")
        temp_matrix_new[1:-1, 1:-1] += self.alpha_matrix[1:-1, 1:-1] * self.dt / self.dx ** 2 * laplacian[1:-1, 1:-1]

        source_sum = np.zeros_like(self.temp_matrix)
        for source in self.sources:
            source_sum += source.get_value(self.X, self.Y, t)

        self.temp_matrix = temp_matrix_new + self.dt * source_sum


if __name__ == "__main__":

    g = Grid(
        width=1,
        height=1,
        step=0.01,
        dt=1e-3,
        nt=500
    )

    # Set initial and boundary temperature
    g.temp_matrix[:] =0
    g.set_boundaries(
        left=0,
        right=0,
        top=0,
        bottom=0,
        boundary_type="dirichlet"
    )

    # Use water properties
    g.create_alpha_matrix(k=0.6, rho=1000, c=4180)


    source_power = 1000  # W/m² (1000 similar to sunlight intensity)
    source_area = np.pi * (0.2)**2  # sigma = 0.02 = 2cm diameter, change area formula depending on source
    total_energy_per_second = source_power * source_area  # in watts (J/s)

    # Compute A so that dT = Q / (m * c), and m = ρ * V
    mass = 1000 * source_area * 0.01  # set depth of grid = 1 cm (only used to model source) , 1000 = density of water
    dT_per_second = total_energy_per_second / (mass * 4180)  # °C/s, 4180 = specific heat capacity of water

    g.add_source(ThermalSource(
        source_type="stationary_gaussian",
        A=dT_per_second ,
        center=(0.5, 0.5),
        sigma=0.02
    ))


    def animate_func_fixed_scale(frame):
        current_time = frame * g.dt  # convert frame count to seconds
        g.update(t=current_time)
        im.set_array(g.temp_matrix)
        return [im]


    fig, ax = plt.subplots()
    im = ax.imshow(
        g.temp_matrix,
        cmap='hot',
        interpolation='nearest',
        vmin=0,
        vmax=100
    )
    plt.colorbar(im, ax=ax)

    ani = animation.FuncAnimation(fig, animate_func_fixed_scale, frames=g.nt, interval=50, blit=True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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

    def create_alpha_matrix(self, default_alpha):
        self.alpha_matrix = default_alpha * np.ones_like(self.temp_matrix)
    def set_alpha_region(self, alpha_region, alpha):
        mask = alpha_region(self.X, self.Y)
        self.alpha_matrix[mask] = alpha

    def add_source(self, source):
        self.sources.append(source)

    def update(self, t):
        for n in range(self.nt):
            temp_matrix_new = self.temp_matrix.copy()

            laplacian = (
                    self.temp_matrix[2:, 1:-1] + self.temp_matrix[:-2, 1:-1] +
                    self.temp_matrix[1:-1, 2:] + self.temp_matrix[1:-1, :-2] -
                    4 * self.temp_matrix[1:-1, 1:-1]
            )


            temp_matrix_new[1:-1, 1:-1] += self.alpha_matrix[1:-1, 1:-1] * self.dt / self.dx ** 2 * laplacian

            source_sum = np.zeros_like(self.temp_matrix)
            for source in self.sources:
                source_sum += source.get_value(self.X, self.Y, t)

            self.temp_matrix = temp_matrix_new + self.dt * source_sum    # Multiply by dt to scale properly

    def animate(self, source_type="none", **kwargs):
        fig, ax = plt.subplots()
        im = ax.imshow(self.temp_matrix, cmap='hot', interpolation='nearest', vmin=np.min(self.temp_matrix), vmax=np.max(self.temp_matrix))
        plt.colorbar(im, ax=ax)

        def animate_func(frame):
            self.update(t=frame)
            im.set_array(self.temp_matrix)
            return [im]

        ani = animation.FuncAnimation(fig, animate_func, frames=self.nt, interval=50, blit=True)
        plt.show()

if __name__ == "__main__":

    g = Grid(
        width=1,
        height=1,
        step=0.01,
        dt=1e-5,
        nt=500
    )

    g.set_boundaries(
        left=0,
        right=0,
        top=0,
        bottom=0,
        boundary_type="dirichlet"
    )

    g.create_alpha_matrix(default_alpha=0.04)
    g.set_alpha_region(alpha_region=lambda X, Y: (X > 0.3) & (X < 0.6) & (Y > 0.3) & (Y < 0.6), alpha=0.01)


    g.add_source(ThermalSource(source_type="stationary_gaussian", A=50, center=(0.5, 0.5), sigma=0.02))
    g.add_source(ThermalSource(source_type="circular_gaussian", omega=2 * np.pi / 60, ))
    g.add_source(ThermalSource(source_type="uniform", A=-0.5))

    g.animate()

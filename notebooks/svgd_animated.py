import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "notebook"


class MultivariateNormal:

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[0]
        self.precision = np.linalg.inv(cov)
        self.chol = np.linalg.cholesky(cov)
        self.logdet = np.log(np.linalg.det(cov))
        self.constant = -0.5 * np.log(2.0 * np.pi) * self.dim

    def log_density(self, x):
        diff = x - self.mean
        return self.constant - 0.5 * self.logdet - 0.5 * diff.T @ self.precision @ diff

    def grad_log_density(self, x):
        diff = x - self.mean
        return -self.precision @ diff

    def sample(self, n=1):
        z = np.random.normal(0, 1, (self.dim, n))
        samples = self.mean.reshape(-1, 1) + self.chol @ z
        return samples.T


class TargetDistributions:
    """Class containing target distributions for SVGD."""

    @staticmethod
    def banana():
        banana_dist = MultivariateNormal(np.array([[0], [4]]).reshape(-1), np.array([[1, 0.5], [0.5, 1]]))

        def log_density(x):
            a, b = 2, 0.2
            y = np.zeros(2)
            y[0] = x[0] / a
            y[1] = x[1] * a + a * b * (x[0] ** 2 + a**2)
            return banana_dist.log_density(y)

        def grad_log_density(x):
            a, b = 2, 0.2
            y = np.zeros(2)
            y[0] = x[0] / a
            y[1] = x[1] * a + a * b * (x[0] ** 2 + a**2)
            grad = banana_dist.grad_log_density(y)
            gradx0 = grad[0] / a + grad[1] * a * b * 2 * x[0]
            gradx1 = grad[1] * a
            return np.array([gradx0, gradx1])

        return log_density, grad_log_density

    @staticmethod
    def donut():
        radius = 2.6
        sigma2 = 0.033

        def log_density(x):
            r = np.linalg.norm(x)
            return -np.power(r - radius, 2) / sigma2

        def grad_log_density(x):
            r = np.linalg.norm(x)
            if r == 0:
                return np.zeros(2)
            grad_x = (x[0] * (radius / r - 1) * 2) / sigma2
            grad_y = (x[1] * (radius / r - 1) * 2) / sigma2
            return np.array([grad_x, grad_y])

        return log_density, grad_log_density

    @staticmethod
    def standard():
        """Bivariate normal distribution."""
        dist = MultivariateNormal(np.zeros(2), np.eye(2))

        def log_density(x):
            return dist.log_density(x)

        def grad_log_density(x):
            return dist.grad_log_density(x)

        return log_density, grad_log_density

    @staticmethod
    def multimodal():
        components = [
            MultivariateNormal(np.array([-1.5, -1.5]), np.eye(2) * 0.8),
            MultivariateNormal(np.array([1.5, 1.5]), np.eye(2) * 0.8),
            MultivariateNormal(np.array([-2, 2]), np.eye(2) * 0.5),
        ]

        def log_density(x):
            densities = [np.exp(comp.log_density(x)) for comp in components]
            return np.log(sum(densities))

        def grad_log_density(x):
            densities = [np.exp(comp.log_density(x)) for comp in components]
            grads = [comp.grad_log_density(x) * d for comp, d in zip(components, densities)]
            return sum(grads) / sum(densities)

        return log_density, grad_log_density

    @staticmethod
    def squiggle():
        squiggle_dist = MultivariateNormal(np.zeros(2), np.array([[2, 0.25], [0.25, 0.5]]))

        def log_density(x):
            y = np.zeros(2)
            y[0] = x[0]
            y[1] = x[1] + np.sin(5 * x[0])
            return squiggle_dist.log_density(y)

        def grad_log_density(x):
            y = np.zeros(2)
            y[0] = x[0]
            y[1] = x[1] + np.sin(5 * x[0])
            grad = squiggle_dist.grad_log_density(y)
            gradx0 = grad[0] + grad[1] * 5 * np.cos(5 * x[0])
            gradx1 = grad[1]
            return np.array([gradx0, gradx1])

        return log_density, grad_log_density


class SVGDAnimatedDemo:
    """SVGD simulation class with animation capabilities. Derived from
    Chi Feng's MCMC Demo (https://github.com/chi-feng/gp-demo)"""

    def __init__(self, target_name="banana", n_particles=200, n_frames=100):
        """Initialize SVGD simulation with chosen target distribution."""
        self.target_distributions = {
            "banana": TargetDistributions.banana(),
            "donut": TargetDistributions.donut(),
            "standard": TargetDistributions.standard(),
            "multimodal": TargetDistributions.multimodal(),
            "squiggle": TargetDistributions.squiggle(),
        }

        self.target_name = target_name
        self.log_density, self.grad_log_density = self.target_distributions[target_name]

        self.n = n_particles
        self.epsilon = 0.01
        self.h = 0.15
        self.use_median = True
        self.use_adagrad = True
        self.alpha = 0.9
        self.fudge_factor = 1e-2

        self.n_frames = n_frames

        self.xmin, self.xmax = -6, 6
        self.ymin, self.ymax = -6, 6

        self.frames_data = []

    def reset(self):
        """Reset the simulation state."""
        self.chain = np.random.normal(0, 1, (self.n, 2))
        self.gradx = np.zeros((self.n, 2))
        self.historical_grad = np.zeros((self.n, 2))
        self.iter = 0

    def compute_density_grid(self, nx=100, ny=100):
        """Compute the target density on a grid."""
        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((ny, nx))

        for i in range(nx):
            for j in range(ny):
                point = np.array([X[j, i], Y[j, i]])
                Z[j, i] = np.exp(self.log_density(point))

        return X, Y, Z

    def step(self):
        """Perform one step of SVGD."""
        n = self.n

        # Precompute gradient of log densities
        for i in range(n):
            self.gradx[i] = self.grad_log_density(self.chain[i])

        # Compute pairwise distances
        dist2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                delta = np.sum(np.power(self.chain[i] - self.chain[j], 2))
                dist2[i, j] = delta
                dist2[j, i] = delta

        if self.use_median:
            flat_dist2 = dist2.flatten()
            median = np.median(flat_dist2)
            self.h = median / np.log(n)

        self.gradx = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                rbf = np.exp(-dist2[i, j] / self.h)
                for k in range(2):
                    grad_rbf = ((self.chain[i, k] - self.chain[j, k]) * 2 * rbf) / self.h
                    self.gradx[i, k] += self.grad_log_density(self.chain[j])[k] * rbf + grad_rbf

            self.gradx[i] /= n

        if self.use_adagrad:
            self.historical_grad = self.alpha * self.historical_grad + (1 - self.alpha) * np.power(self.gradx, 2)
            self.gradx /= self.fudge_factor + np.sqrt(self.historical_grad)

        self.gradx *= self.epsilon

        self.chain += self.gradx

        self.iter += 1

        return self.chain.copy(), self.gradx.copy(), self.h

    def compute_hist(self, bins=20):
        """Compute histograms for marginal distributions."""
        x_hist, x_edges = np.histogram(self.chain[:, 0], bins=bins, range=(self.xmin, self.xmax), density=True)
        y_hist, y_edges = np.histogram(self.chain[:, 1], bins=bins, range=(self.ymin, self.ymax), density=True)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        return x_centers, x_hist, y_centers, y_hist

    def run_simulation(self):
        """Run the simulation for n_frames steps and store the frames."""

        self.reset()

        X, Y, Z = self.compute_density_grid()

        self.frames_data = []
        self.frames_data.append(
            {
                "iter": self.iter,
                "particles": self.chain.copy(),
                "gradients": self.gradx.copy(),
                "h": self.h,
                "histograms": self.compute_hist(bins=30),
            }
        )

        for _ in range(self.n_frames):
            _, _, _ = self.step()
            self.frames_data.append(
                {
                    "iter": self.iter,
                    "particles": self.chain.copy(),
                    "gradients": self.gradx.copy(),
                    "h": self.h,
                    "histograms": self.compute_hist(bins=30),
                }
            )

    def create_animation(self):
        """Create an animated plot of the SVGD simulation."""

        self.run_simulation()

        X, Y, Z = self.compute_density_grid()

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            specs=[[{"colspan": 2}, None], [{"type": "xy"}, {"type": "xy"}]],
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        fig.add_trace(
            go.Contour(
                x=np.linspace(self.xmin, self.xmax, Z.shape[1]),
                y=np.linspace(self.ymin, self.ymax, Z.shape[0]),
                z=Z,
                colorscale="Blues",
                showscale=False,
                opacity=0.8,
                contours=dict(start=0, end=Z.max(), size=(Z.max() - 0) / 10),
                line=dict(width=0.5),
            ),
            row=2,
            col=1,
        )

        init_frame = self.frames_data[0]

        particles = fig.add_trace(
            go.Scatter(
                x=init_frame["particles"][:, 0],
                y=init_frame["particles"][:, 1],
                mode="markers",
                marker=dict(color="black", size=5, opacity=0.6),
                name="Particles",
            ),
            row=2,
            col=1,
        )

        init_gradx = init_frame["gradients"]
        norm = np.linalg.norm(init_gradx, axis=1)
        max_norm = max(norm.max(), 1e-10)
        scale = 0.5 / max_norm

        arrow_x = []
        arrow_y = []

        for i in range(self.n):
            if norm[i] > 1e-4 * max_norm:
                xi, yi = init_frame["particles"][i]
                dx, dy = init_gradx[i] * scale
                arrow_x.extend([xi, xi + dx, None])
                arrow_y.extend([yi, yi + dy, None])

        vectors = fig.add_trace(
            go.Scatter(
                x=arrow_x, y=arrow_y, mode="lines", line=dict(color="rgba(0,0,0,0.5)", width=1), name="Gradients"
            ),
            row=2,
            col=1,
        )

        x_centers, x_hist, y_centers, y_hist = init_frame["histograms"]

        x_histogram = fig.add_trace(
            go.Bar(
                x=x_centers,
                y=x_hist,
                marker=dict(color="rgba(102, 153, 187, 0.6)"),
                width=(x_centers[1] - x_centers[0]) * 0.9,
                name="X Histogram",
            ),
            row=1,
            col=1,
        )

        y_histogram = fig.add_trace(
            go.Bar(
                x=y_hist,
                y=y_centers,
                orientation="h",
                marker=dict(color="rgba(102, 153, 187, 0.6)"),
                width=(y_centers[1] - y_centers[0]) * 0.9,
                name="Y Histogram",
            ),
            row=2,
            col=2,
        )

        frames = []
        for i, frame_data in enumerate(self.frames_data):
            gradx = frame_data["gradients"]
            norm = np.linalg.norm(gradx, axis=1)
            max_norm = max(norm.max(), 1e-10)
            scale = 0.5 / max_norm

            arrow_x = []
            arrow_y = []

            for j in range(self.n):
                if norm[j] > 1e-4 * max_norm:
                    xj, yj = frame_data["particles"][j]
                    dx, dy = gradx[j] * scale
                    arrow_x.extend([xj, xj + dx, None])
                    arrow_y.extend([yj, yj + dy, None])

            x_centers, x_hist, y_centers, y_hist = frame_data["histograms"]

            frame = go.Frame(
                data=[
                    fig.data[0],

                    go.Scatter(
                        x=frame_data["particles"][:, 0],
                        y=frame_data["particles"][:, 1],
                        mode="markers",
                        marker=dict(color="black", size=5, opacity=0.6),
                    ),

                    go.Scatter(x=arrow_x, y=arrow_y, mode="lines", line=dict(color="rgba(0,0,0,0.5)", width=1)),

                    go.Bar(
                        x=x_centers,
                        y=x_hist,
                        marker=dict(color="rgba(102, 153, 187, 0.6)"),
                        width=(x_centers[1] - x_centers[0]) * 0.9,
                    ),

                    go.Bar(
                        x=y_hist,
                        y=y_centers,
                        orientation="h",
                        marker=dict(color="rgba(102, 153, 187, 0.6)"),
                        width=(y_centers[1] - y_centers[0]) * 0.9,
                    ),
                ],
                name=f"frame{i}",
                layout=go.Layout(
                    title_text=f"SVGD: {self.target_name.title()} Distribution (iter: {frame_data['iter']}, bandwidth: {frame_data['h']:.3f})"
                ),
            )
            frames.append(frame)

        fig.frames = frames

        fig.update_xaxes(range=[self.xmin, self.xmax], row=2, col=1)
        fig.update_yaxes(range=[self.ymin, self.ymax], row=2, col=1)
        fig.update_xaxes(range=[self.xmin, self.xmax], row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=2)
        fig.update_yaxes(range=[self.ymin, self.ymax], row=2, col=2)

        fig.update_layout(
            height=700,
            width=800,
            title_text=f"SVGD: {self.target_name.title()} Distribution (iter: {self.frames_data[0]['iter']}, bandwidth: {self.frames_data[0]['h']:.3f})",
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=30, b=10),

            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 16},
                        "prefix": "Iteration: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame{i}"],
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": str(frame_data["iter"]),
                            "method": "animate",
                        }
                        for i, frame_data in enumerate(self.frames_data)
                    ],
                }
            ],
        )

        return fig


def create_svgd_demo(target: str = "banana", n_particles: int = 200, n_frames: int = 100) -> plotly.graph_objects.Figure:
    """Create and return an animated SVGD demo.
    
    Parameters
    ----------
    target : str, default="banana"
        Name of the target distribution to use. Options include 
        "banana", "donut", "standard", "multimodal", and "squiggle".
    n_particles : int, default=200
        Number of particles to use in the simulation.
    n_frames : int, default=100
        Number of animation frames to generate.
        
    Returns
    -------
    plotly.graph_objects.Figure
        A plotly Figure object containing the animated SVGD demonstration.
    """
    demo = SVGDAnimatedDemo(target_name=target, n_particles=n_particles, n_frames=n_frames)
    return demo.create_animation()


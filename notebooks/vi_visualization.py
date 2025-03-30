import numpy as np
import scipy.stats as stats
import scipy.special
import panel as pn
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, Dropdown
import plotly.io as pio

pn.extension('plotly')

target_distributions = {}

def mix_gauss_pdf(z, w1=0.5, mu1=-3.0, s1=1.0, mu2=3.0, s2=1.5):
    """PDF of a mixture of two Gaussians."""
    return w1 * stats.norm.pdf(z, mu1, s1) + (1 - w1) * stats.norm.pdf(z, mu2, s2)

def mix_gauss_logpdf(z, w1=0.5, mu1=-3.0, s1=1.0, mu2=3.0, s2=1.5):
    """Log-PDF of a mixture of two Gaussians using logsumexp for stability."""
    log_comp1 = np.log(w1) + stats.norm.logpdf(z, mu1, s1)
    log_comp2 = np.log(1 - w1) + stats.norm.logpdf(z, mu2, s2)
    log_probs = np.vstack([log_comp1, log_comp2])
    return scipy.special.logsumexp(log_probs, axis=0)

target_distributions['Mixture Gaussian'] = {
    'pdf': mix_gauss_pdf,
    'logpdf': mix_gauss_logpdf,
    'params': {'w1': 0.5, 'mu1': -3.0, 's1': 1.0, 'mu2': 3.0, 's2': 1.5},
    'z_range': (-10, 10)
}

def gumbel_pdf(z, loc=0, scale=1):
    """PDF of the Gumbel distribution."""
    return stats.gumbel_r.pdf(z, loc=loc, scale=scale)

def gumbel_logpdf(z, loc=0, scale=1):
    """Log-PDF of the Gumbel distribution."""
    return stats.gumbel_r.logpdf(z, loc=loc, scale=scale)

target_distributions['Gumbel'] = {
    'pdf': gumbel_pdf,
    'logpdf': gumbel_logpdf,
    'params': {'loc': 2.0, 'scale': 2.0},
    'z_range': (-5, 15)
}

def student_t_pdf(z, df=3, loc=0, scale=1):
    """PDF of the Student's t-distribution."""
    return stats.t.pdf(z, df=df, loc=loc, scale=scale)

def student_t_logpdf(z, df=3, loc=0, scale=1):
    """Log-PDF of the Student's t-distribution."""
    return stats.t.logpdf(z, df=df, loc=loc, scale=scale)

target_distributions['Student-t'] = {
    'pdf': student_t_pdf,
    'logpdf': student_t_logpdf,
    'params': {'df': 3, 'loc': -2, 'scale': 1},
    'z_range': (-10, 10)
}

def gaussian_pdf(z, mu, sigma):
    """Gaussian PDF."""
    sigma = max(sigma, 1e-9)
    return stats.norm.pdf(z, mu, sigma)

def gaussian_logpdf(z, mu, sigma):
    """Gaussian log-PDF."""
    sigma = max(sigma, 1e-9)
    return stats.norm.logpdf(z, mu, sigma)

def calculate_elbo(target_logpdf, target_params, mu, sigma, n_samples=2000):
    """Calculate ELBO (E_q[log p(z) - log q(z)]) using Monte Carlo sampling."""
    sigma = max(sigma, 1e-9)
    samples_q = np.random.normal(mu, sigma, n_samples)

    log_p_samples = target_logpdf(samples_q, **target_params)
    log_q_samples = gaussian_logpdf(samples_q, mu, sigma)

    # Replace NaNs and ensure -inf propagation
    log_p_samples = np.nan_to_num(log_p_samples, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    log_q_samples = np.nan_to_num(log_q_samples, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    diff = log_p_samples - log_q_samples

    # If p=0 where q>0 for any sample, ELBO should be -inf
    # This corresponds to diff being -inf if log_q is finite
    if np.any(diff == -np.inf):
         finite_q = log_q_samples != np.inf
         if np.any(diff[finite_q] == -np.inf):
             return -np.inf

    # Calculate mean over finite values, ignoring potential +inf or -inf diffs
    # (though -inf case handled above, +inf shouldn't strictly happen sampling from q)
    finite_diff = diff[np.isfinite(diff)]
    if len(finite_diff) == 0:
        if np.any((log_p_samples == -np.inf) & (log_q_samples > -np.inf)):
            return -np.inf
        return np.nan 

    elbo = np.mean(finite_diff)
    return elbo

def calculate_kl_divergence(target_logpdf, target_params, mu, sigma, n_samples=2000):
    """Calculate KL(q || p) = E_q[log q(z) - log p(z)] using Monte Carlo sampling."""
    sigma = max(sigma, 1e-9)
    samples_q = np.random.normal(mu, sigma, n_samples)

    log_p_samples = target_logpdf(samples_q, **target_params)
    log_q_samples = gaussian_logpdf(samples_q, mu, sigma)

    # Replace NaNs and ensure -inf propagation
    log_p_samples = np.nan_to_num(log_p_samples, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    log_q_samples = np.nan_to_num(log_q_samples, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    diff = log_q_samples - log_p_samples

    # If p=0 where q>0 for any sample, KL divergence is +inf
    # This corresponds to diff being +inf if log_q is finite
    if np.any(diff == np.inf):
         finite_q = log_q_samples != np.inf
         if np.any(diff[finite_q] == np.inf):
             return np.inf

    finite_diff = diff[np.isfinite(diff)]
    if len(finite_diff) == 0:
        # Check if it was due to the infinite KL case handled above
        if np.any((log_p_samples == -np.inf) & (log_q_samples > -np.inf)):
            return np.inf
        return np.nan 

    kl_div = np.mean(finite_diff)
    return kl_div


class VariationalInferenceViz:
    """
    An interactive visualization for 1D Variational Inference.
    Uses pn.bind for dynamic updates. The metric (ELBO or KL) is fixed on instantiation.
    """
    def __init__(self, n_samples=2000, metric='KL'):
        """
        Initializes the visualization components.

        Args:
            n_samples (int): Number of samples for Monte Carlo estimation.
            metric (str): The metric to display ('ELBO' or 'KL'). Defaults to 'KL'.
        """
        if metric not in ['ELBO', 'KL']:
            raise ValueError("Metric must be 'ELBO' or 'KL'")
        self.n_samples = n_samples
        self.metric = metric 

        self.mu_slider = pn.widgets.FloatSlider(
            name='Mean (μ)', start=-10.0, end=10.0, step=0.1, value=0.0
        )
        self.sigma_slider = pn.widgets.FloatSlider(
            name='Std Dev (σ)', start=0.1, end=10.0, step=0.1, value=1.0
        )
        self.target_selector = pn.widgets.Select(
            name='Target Distribution', options=list(target_distributions.keys()), width=180, value='Student-t'
        )

        initial_fig = self._create_plotly_fig(
            self.mu_slider.value,
            self.sigma_slider.value,
            self.target_selector.value
        )
        self.plot_pane = pn.pane.Plotly(initial_fig, width=700, height=400)

        self.plot_pane.param.update(
            object=pn.bind(
                self._update_plot,
                mu=self.mu_slider,
                sigma=self.sigma_slider,
                target_name=self.target_selector
            )
        )

    def _create_plotly_fig(self, mu, sigma, target_name):
        """Generates the Plotly figure object based on current parameters."""
        target_info = target_distributions[target_name]
        target_pdf = target_info['pdf']
        target_logpdf = target_info['logpdf']
        target_params = target_info['params']
        z_min_hint, z_max_hint = target_info['z_range']

        sigma = max(sigma, 1e-9)

        plot_z_min = min(z_min_hint, mu - 4 * sigma)
        plot_z_max = max(z_max_hint, mu + 4 * sigma)
        if plot_z_max <= plot_z_min:
            plot_z_max = plot_z_min + 1.0
        z = np.linspace(plot_z_min, plot_z_max, 500)

        p_values = target_pdf(z, **target_params)
        q_values = gaussian_pdf(z, mu, sigma)

        metric_label = self.metric 
        if self.metric == 'KL':
            metric_value = calculate_kl_divergence(target_logpdf, target_params, mu, sigma, n_samples=self.n_samples)
        else: 
            metric_value = calculate_elbo(target_logpdf, target_params, mu, sigma, n_samples=self.n_samples)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=z, y=p_values, mode='lines', name=f'Target: {target_name}',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=z, y=q_values, mode='lines', name=f'Approx q(z): N({mu:.2f}, {sigma:.2f}²)',
            line=dict(color='red', width=2, dash='dash')
        ))

        if np.isinf(metric_value):
            metric_text = f"{metric_label} ≈ {metric_value:+.1f}" 
        elif np.isnan(metric_value):
            metric_text = f"{metric_label} = NaN" 
        else:
            metric_text = f"{metric_label} ≈ {metric_value:.4f}"

        fig.update_layout(
            title="Variational Inference: Target p(z) vs Approx q(z)",
            xaxis_title="z",
            yaxis_title="Density",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=20, r=20, t=50, b=20),
            width=700, height=400,
            annotations=[
                dict(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=metric_text, 
                    showarrow=False,
                    font=dict(size=16, color="black"), align="left",
                    bgcolor="#e8f0fe", bordercolor="#a9c7e8",
                    borderwidth=1, borderpad=8,
                    xanchor="left", yanchor="top"
                )
            ]
        )
        return fig

    def _update_plot(self, mu, sigma, target_name):
        """Creates the Plotly figure based on widget values."""
        fig = self._create_plotly_fig(mu, sigma, target_name)
        return fig

    @property
    def layout(self):
        """Returns the Panel layout object for the visualization."""
        opt_goal = "minimize" if self.metric == 'KL' else "maximize"
        description_text = f"Select a target distribution and adjust the parameters (μ, σ) of the approximating Gaussian distribution (red dashed line) to {opt_goal} the {self.metric}."

        widget_row = pn.Row(
            self.target_selector,
            pn.Column(self.mu_slider, self.sigma_slider)
        )

        return pn.Column(
            "## Interactive Variational Inference Demo",
            pn.pane.Markdown(
                description_text,
                max_width=680,
                styles={'overflow-wrap': 'break-word'}
            ),
            widget_row,
            self.plot_pane,
            name="VI Visualization"
        )

class JupyterVariationalInferenceViz:
    """
    A Jupyter notebook-compatible interactive visualization for 1D Variational Inference.
    Uses ipywidgets for better compatibility with Jupyter environments.
    """
    def __init__(self, n_samples=2000, metric='KL'):
        """
        Initializes the Jupyter-compatible visualization components.

        Args:
            n_samples (int): Number of samples for Monte Carlo estimation.
            metric (str): The metric to display ('ELBO' or 'KL'). Defaults to 'KL'.
        """
        if metric not in ['ELBO', 'KL']:
            raise ValueError("Metric must be 'ELBO' or 'KL'")
        self.n_samples = n_samples
        self.metric = metric
        
        self.fig = self._create_plotly_fig(
            mu=0.0,
            sigma=1.0,
            target_name='Student-t'
        )
        
        self.target_dropdown = Dropdown(
            options=list(target_distributions.keys()),
            value='Student-t',
            description='Target:',
            style={'description_width': 'initial'}
        )
        
        self.mu_slider = FloatSlider(
            value=0.0,
            min=-10.0,
            max=10.0,
            step=0.1,
            description='Mean (μ):',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.sigma_slider = FloatSlider(
            value=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std Dev (σ):',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
    def _create_plotly_fig(self, mu, sigma, target_name):
        """Generates the Plotly figure object based on current parameters."""
        target_info = target_distributions[target_name]
        target_pdf = target_info['pdf']
        target_logpdf = target_info['logpdf']
        target_params = target_info['params']
        z_min_hint, z_max_hint = target_info['z_range']

        sigma = max(sigma, 1e-9)

        plot_z_min = min(z_min_hint, mu - 4 * sigma)
        plot_z_max = max(z_max_hint, mu + 4 * sigma)
        if plot_z_max <= plot_z_min:
            plot_z_max = plot_z_min + 1.0
        z = np.linspace(plot_z_min, plot_z_max, 500)

        p_values = target_pdf(z, **target_params)
        q_values = stats.norm.pdf(z, mu, sigma)

        if self.metric == 'KL':
            metric_value = calculate_kl_divergence(target_logpdf, target_params, mu, sigma, n_samples=self.n_samples)
        else:
            metric_value = calculate_elbo(target_logpdf, target_params, mu, sigma, n_samples=self.n_samples)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=z, y=p_values, mode='lines', name=f'Target: {target_name}',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=z, y=q_values, mode='lines', name=f'Approx q(z): N({mu:.2f}, {sigma:.2f}²)',
            line=dict(color='red', width=2, dash='dash')
        ))

        if np.isinf(metric_value):
            metric_text = f"{self.metric} ≈ {metric_value:+.1f}" # Show +inf or -inf
        elif np.isnan(metric_value):
            metric_text = f"{self.metric} = NaN" # Indicate undefined
        else:
            metric_text = f"{self.metric} ≈ {metric_value:.4f}"

        fig.update_layout(
            title="Variational Inference: Target p(z) vs Approx q(z)",
            xaxis_title="z",
            yaxis_title="Density",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=20, r=20, t=50, b=20),
            width=700, height=400,
            annotations=[
                dict(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=metric_text,
                    showarrow=False,
                    font=dict(size=16, color="black"), align="left",
                    bgcolor="#e8f0fe", bordercolor="#a9c7e8",
                    borderwidth=1, borderpad=8,
                    xanchor="left", yanchor="top"
                )
            ]
        )
        return fig
        
    def _update_plot(self, mu, sigma, target_name):
        """Creates an updated Plotly figure based on widget values."""
        return self._create_plotly_fig(mu, sigma, target_name)
    
    def show(self):
        """Display the interactive visualization."""
        pio.renderers.default = "notebook"
        
        opt_goal = "minimize" if self.metric == 'KL' else "maximize"
        description = f"<h3>Interactive Variational Inference Demo</h3>"
        description += f"<p>Select a target distribution and adjust the parameters (μ, σ) of the approximating Gaussian distribution (red dashed line) to {opt_goal} the {self.metric}.</p>"
        from IPython.display import HTML, display
        display(HTML(description))
        
        def update(target=self.target_dropdown.value, 
                  mu=self.mu_slider.value, 
                  sigma=self.sigma_slider.value):
            self.fig = self._update_plot(mu, sigma, target)
            return self.fig
        
        return interact(update, 
                       target=self.target_dropdown, 
                       mu=self.mu_slider, 
                       sigma=self.sigma_slider)

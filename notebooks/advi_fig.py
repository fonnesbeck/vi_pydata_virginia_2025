import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "(a) Latent variable space",
        "(b) Real coordinate space", 
        "(c) Standardized space"
    )
)

prior_color = 'grey'
posterior_color = 'purple'
approx_color = 'green'
prior_dash = 'dot'
posterior_dash = 'solid'
approx_dash = 'solid'

x_a = np.linspace(0, 3.5, 300)
prior_a = 0.4 * stats.gamma.pdf(x_a, a=1.5, loc=0.1, scale=1.0)
posterior_a = stats.norm.pdf(x_a, loc=1.8, scale=0.4) * 0.9
approx_a = stats.norm.pdf(x_a, loc=1.9, scale=0.35) * 0.95

x_b = np.linspace(-1.5, 2.5, 300)
prior_b = 0.6 * stats.norm.pdf(x_b, loc=-0.2, scale=0.6)
posterior_b = stats.norm.pdf(x_b, loc=0.5, scale=0.3)
approx_b = stats.norm.pdf(x_b, loc=0.5, scale=0.35)

x_c = np.linspace(-3, 3, 300)
prior_c = 0.18 * stats.norm.pdf(x_c, loc=0, scale=1.5)
posterior_c = stats.norm.pdf(x_c, loc=0, scale=0.8)
approx_c = stats.norm.pdf(x_c, loc=0, scale=1.0)

fig.add_trace(go.Scatter(x=x_a, y=prior_a, mode='lines', name='Prior', line=dict(color=prior_color, dash=prior_dash), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=x_a, y=posterior_a, mode='lines', name='Posterior', line=dict(color=posterior_color, dash=posterior_dash), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=x_a, y=approx_a, mode='lines', name='Approximation', line=dict(color=approx_color, dash=approx_dash), showlegend=True), row=1, col=1)

fig.add_trace(go.Scatter(x=x_b, y=prior_b, mode='lines', name='Prior', line=dict(color=prior_color, dash=prior_dash), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_b, y=posterior_b, mode='lines', name='Posterior', line=dict(color=posterior_color, dash=posterior_dash), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=x_b, y=approx_b, mode='lines', name='Approximation', line=dict(color=approx_color, dash=approx_dash), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=x_c, y=prior_c, mode='lines', name='Prior', line=dict(color=prior_color, dash=prior_dash), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=x_c, y=posterior_c, mode='lines', name='Posterior', line=dict(color=posterior_color, dash=posterior_dash), showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=x_c, y=approx_c, mode='lines', name='Approximation', line=dict(color=approx_color, dash=approx_dash), showlegend=False), row=1, col=3)

y_axis_max = 1.2

fig.update_layout(
    height=400,
    width=900,
    legend_title_text='Density Type',
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.98
    ),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.update_xaxes(title_text="θ", range=[x_a.min(), x_a.max()], row=1, col=1)
fig.update_xaxes(title_text="ζ", range=[x_b.min(), x_b.max()], row=1, col=2)
fig.update_xaxes(title_text="η", range=[x_c.min(), x_c.max()], row=1, col=3)

fig.update_yaxes(title_text="Density", range=[0, y_axis_max], row=1, col=1)
fig.update_yaxes(range=[0, y_axis_max], showticklabels=True, row=1, col=2)
fig.update_yaxes(range=[0, y_axis_max], showticklabels=True, row=1, col=3)

arrow_length = 40
arrow_size = 1.5
arrow_width = 1.5
y_top = 0.80
y_bottom = 0.60
x_pos1 = 0.33
x_pos2 = 0.66

fig.add_annotation(
    xref="paper", yref="paper",
    x=x_pos1, y=y_top, text="T",
    showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width,
    ax=-arrow_length, ay=0
)
fig.add_annotation(
    xref="paper", yref="paper",
    x=x_pos1, y=y_bottom, text="T<sup>-1</sup>",
    showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width,
    ax=arrow_length, ay=0
)

fig.add_annotation(
    xref="paper", yref="paper",
    x=x_pos2, y=y_top, text="S<sub>μ,ω</sub>",
    showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width,
    ax=-arrow_length, ay=0
)
fig.add_annotation(
    xref="paper", yref="paper",
    x=x_pos2, y=y_bottom, text="S<sup>-1</sup><sub>μ,ω</sub>",
    showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=arrow_width,
    ax=arrow_length, ay=0
)

output_dir = "images"
output_file = os.path.join(output_dir, "advi.png")

os.makedirs(output_dir, exist_ok=True)

fig.write_image(output_file)

print(f"Figure saved to {output_file}")
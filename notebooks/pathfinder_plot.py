import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.stats import multivariate_normal, norm, bernoulli
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
import os

# Simulate data for logistic regression
np.random.seed(123)
N_samples = 100
N_features = 2

beta_true = np.array([-1.0, 2.0])
X = np.random.randn(N_samples, N_features - 1)
X_design = np.hstack([np.ones((N_samples, 1)), X])

linear_pred = X_design @ beta_true
prob = 1 / (1 + np.exp(-linear_pred))
y = stats.bernoulli.rvs(p=prob)

def neg_log_posterior(beta, X, y, prior_std_dev=1.0):
    """Calculates the negative log-posterior for logistic regression."""
    if np.any(np.isinf(beta)) or np.any(np.isnan(beta)):
        return np.inf

    linear_pred = X @ beta
    log_likelihood = np.sum(y * linear_pred - np.logaddexp(0, linear_pred))
    log_prior = np.sum(norm.logpdf(beta, loc=0, scale=prior_std_dev))
    nlp = -(log_likelihood + log_prior)

    if np.isnan(nlp) or np.isinf(nlp):
        return np.inf

    return nlp

def neg_log_posterior_gradient(beta, X, y, prior_std_dev=1.0):
    """Calculates the gradient of the negative log-posterior."""
    beta = np.asarray(beta, dtype=float)

    if np.any(np.abs(beta) > 50):
        return (beta / (prior_std_dev**2)) * 1e3

    linear_pred = X @ beta
    stable_linear_pred = np.clip(linear_pred, -700, 700)
    prob = 1 / (1 + np.exp(-stable_linear_pred))

    grad_neg_log_likelihood = X.T @ (prob - y)
    grad_neg_log_prior = beta / (prior_std_dev**2)
    grad = grad_neg_log_likelihood + grad_neg_log_prior

    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
        print(f"Warning: NaN/Inf detected in gradient for beta={beta}. Returning prior gradient.")
        grad = beta / (prior_std_dev**2)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            print("Warning: Prior gradient also NaN/Inf. Returning zero gradient.")
            grad = np.zeros_like(beta)

    return grad

optimization_path = []
gradients_path = []

def store_path_callback(intermediate_result):
    """Callback function to store optimization path and gradients."""
    if hasattr(intermediate_result, 'x'):
        current_beta = intermediate_result.x
    else:
        current_beta = intermediate_result

    if not optimization_path or not np.allclose(current_beta, optimization_path[-1]):
        optimization_path.append(np.copy(current_beta))
        current_grad = neg_log_posterior_gradient(current_beta, X_design, y)
        gradients_path.append(current_grad)

beta_initial = np.array([-0.6, 0.2])
optimization_path.append(np.copy(beta_initial))
gradients_path.append(neg_log_posterior_gradient(beta_initial, X_design, y))

print(f"Starting L-BFGS from: {beta_initial}")

result = minimize(
    neg_log_posterior,
    beta_initial,
    args=(X_design, y),
    method='L-BFGS-B',
    jac=neg_log_posterior_gradient,
    callback=store_path_callback,
    options={'maxiter': 20, 'disp': True, 'gtol': 1e-5}
)

if not np.allclose(result.x, optimization_path[-1]):
    optimization_path.append(result.x)
    gradients_path.append(neg_log_posterior_gradient(result.x, X_design, y))

print(f"\nOptimization finished. Found mode (MAP estimate): {result.x}")
print(f"Status: {result.message}")
print(f"Total points in path: {len(optimization_path)}")

indices_to_approximate = list(range(1, min(8, len(optimization_path))))
approximations = []

print("\nCalculating approximations at selected path points:")

def grad_for_hessian(beta, X, y):
    return neg_log_posterior_gradient(beta, X, y)

for i in indices_to_approximate:
    beta_l = optimization_path[i]
    if i >= len(gradients_path):
        print(f"Warning: Gradient not found for path index {i}. Skipping approximation.")
        continue
    grad_nlp_l = gradients_path[i]

    print(f"  Point {i}: beta = {np.round(beta_l, 3)}")

    epsilon = np.sqrt(np.finfo(float).eps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            hessian_nlp_l = approx_fprime(beta_l, grad_for_hessian, epsilon*np.ones_like(beta_l), X_design, y)
            if np.any(np.isnan(hessian_nlp_l)) or np.any(np.isinf(hessian_nlp_l)):
                raise ValueError("NaN/Inf in computed Hessian")
        except Exception as e:
            print(f"    Warning: Hessian calculation failed at point {i}: {e}")
            continue

    try:
        ridge = 1e-6
        cov_l = np.linalg.inv(hessian_nlp_l + ridge * np.identity(N_features))
        cov_l = (cov_l + cov_l.T) / 2.0
        np.linalg.cholesky(cov_l)
    except np.linalg.LinAlgError:
        print(f"    Warning: Hessian at point {i} not positive definite or invertible. Skipping.")
        continue
    except ValueError as e:
        print(f"    Warning: Skipping inverse due to invalid Hessian at point {i}: {e}")
        continue

    grad_logp_l = -grad_nlp_l
    mean_l = beta_l + cov_l @ grad_logp_l

    approximations.append({
        'path_index': i,
        'beta_l': beta_l,
        'mean_approx': mean_l,
        'cov_approx': cov_l
    })
    print(f"    Mean Approx (mu_{i}): {np.round(mean_l, 3)}")
    print(f"    Cov Approx (Sigma_{i}): \n{np.round(cov_l, 3)}")

print(f"\nGenerated {len(approximations)} normal approximations along the path.")
print("Stored in the 'approximations' list as dictionaries containing:")
print("  'path_index', 'beta_l', 'mean_approx', 'cov_approx'")

def plot_ellipse(ax, mean, cov, confidence=0.95, **kwargs):
    """Plots an ellipse representing the covariance matrix."""
    if np.isscalar(cov) or cov.shape != (2, 2):
        print(f"Warning: Cannot plot ellipse for cov shape {cov.shape}.")
        return None
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if np.any(eigenvalues <= 0):
            print("Warning: Covariance matrix not positive definite, cannot plot ellipse reliably.")
            eigenvalues = np.maximum(eigenvalues, 1e-9)
    except np.linalg.LinAlgError:
        print("Warning: Eigendecomposition failed for covariance matrix. Cannot plot ellipse.")
        return None

    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    scale_factor = np.sqrt(stats.chi2.isf(1 - confidence, df=2))
    width, height = 2 * scale_factor * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor='none', **kwargs)
    return ax.add_patch(ellipse)

path_array = np.array(optimization_path)
beta0_min, beta0_max = min(path_array[:,0]) - 1, max(path_array[:,0]) + 1
beta1_min, beta1_max = min(path_array[:,1]) - 1, max(path_array[:,1]) + 1
beta0_range = np.linspace(beta0_min, beta0_max, 100)
beta1_range = np.linspace(beta1_min, beta1_max, 100)
B0, B1 = np.meshgrid(beta0_range, beta1_range)
Z = np.zeros_like(B0)
for i_b0 in range(B0.shape[0]):
    for i_b1 in range(B0.shape[1]):
        Z[i_b0, i_b1] = neg_log_posterior(np.array([B0[i_b0, i_b1], B1[i_b0, i_b1]]), X_design, y)

# Handle potential issues with Z for contour plotting
if not np.all(np.isfinite(Z)):
    print("Warning: Non-finite values found in contour data. Replacing with boundary values.")
    z_finite = Z[np.isfinite(Z)]
    if len(z_finite) > 0:
        z_max_finite = np.nanmax(z_finite)
        z_min_finite = np.nanmin(z_finite)
        Z = np.nan_to_num(Z, nan=z_max_finite, posinf=z_max_finite, neginf=z_min_finite)
    else:
        print("Warning: All contour data is non-finite. Cannot plot contours.")
        Z = None # Flag to skip contour plotting

contour_levels = None
if Z is not None:
    min_z, max_z = np.min(Z), np.max(Z)
    if min_z < max_z: # Ensure valid range for logspace
        # Use logspace for better visualization near minimum, add small epsilon for stability
        contour_levels = np.logspace(np.log10(min_z + 1e-9), np.log10(max_z + 1e-9), 15)
    else:
        print("Warning: Contour range is invalid (min >= max). Skipping contour plot.")
        Z = None # Flag to skip contour plotting

def setup_plot(ax, title):
    """Sets up common plot elements."""
    if Z is not None and contour_levels is not None:
        # Plot contours with lower zorder and alpha
        ax.contour(B0, B1, Z, levels=contour_levels, cmap="viridis", alpha=0.5, linewidths=1.0, zorder=1)
        # Add finer contours for better detail
        ax.contour(B0, B1, Z, levels=30, cmap="viridis", alpha=0.2, linewidths=0.5, zorder=1)
    ax.set_title(title)
    ax.set_xlabel("Beta 0 (Intercept)")
    ax.set_ylabel("Beta 1")
    ax.set_xlim(beta0_min, beta0_max)
    ax.set_ylim(beta1_min, beta1_max)
    ax.grid(True, alpha=0.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1 row, 3 columns

# --- Plot 1: Optimization Path (on axes[0]) ---
ax1 = axes[0]
setup_plot(ax1, "Optimization Path")
ax1.plot(path_array[:, 0], path_array[:, 1], 'k.-', label='Optimization Path', zorder=3)
ax1.scatter(path_array[0, 0], path_array[0, 1], color='red', s=100, zorder=4, label='Start')
ax1.scatter(path_array[-1, 0], path_array[-1, 1], color='lime', s=100, zorder=4, label='End (MAP)')
ax1.legend()

# --- Plot 2: Normal Approximations (Ellipses) (on axes[1]) ---
ax2 = axes[1]
setup_plot(ax2, "Pathfinder Normal Approximations")
# Use a single color and thicker lines for ellipses
ellipse_color = 'red'
ellipse_linewidth = 2.0
for idx, approx in enumerate(approximations):
    if approx['cov_approx'].shape == (N_features, N_features) and N_features==2:
        plot_ellipse(ax2, approx['mean_approx'], approx['cov_approx'],
                     edgecolor=ellipse_color, linewidth=ellipse_linewidth, zorder=4)
                     # No individual labels for clarity
    else:
        print(f"Skipping ellipse plot for point {approx['path_index']} (dim != 2 or invalid cov)")


# --- Plot 3: Samples from Approximations (on axes[2]) ---
ax3 = axes[2]
setup_plot(ax3, "Samples from Pathfinder Approximations")
n_samples_per_approx = 50 

sample_colors = plt.cm.plasma(np.linspace(0, 1, len(approximations)))

for idx, approx in enumerate(approximations):
    mean_l = approx['mean_approx']
    cov_l = approx['cov_approx']
    path_idx = approx['path_index']

    if cov_l.shape != (N_features, N_features) or N_features != 2:
        print(f"Skipping sampling for point {path_idx} (dim != 2 or invalid cov shape)")
        continue

    try:
        # Ensure covariance is positive semi-definite for sampling
        cov_l_reg = cov_l + np.eye(N_features) * 1e-9
        samples = multivariate_normal.rvs(mean=mean_l, cov=cov_l_reg, size=n_samples_per_approx)
        ax3.scatter(samples[:, 0], samples[:, 1], color=sample_colors[idx], s=10, alpha=0.6, zorder=3,
                    label=f"Samples (Approx {path_idx})" if idx == 0 else "") 
    except np.linalg.LinAlgError:
        print(f"Warning: Covariance matrix at point {path_idx} not positive definite for sampling. Skipping.")
    except ValueError as e:
         print(f"Warning: Sampling failed for point {path_idx}: {e}. Skipping.")


plt.tight_layout()

output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'pathfinder.png')

try:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
except Exception as e:
    print(f"\nError saving plot: {e}")
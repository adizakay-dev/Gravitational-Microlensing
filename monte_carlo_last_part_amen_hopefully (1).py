import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.lines import Line2D

# =================================================================
# 1. USER SETTINGS: BASELINE & FILE
# =================================================================
I_STAR = 16.735  # <--- TYPE YOUR MAGNITUDE WITHOUT EVENT HERE
FILE_PATH = 'ADD PATH'

# =================================================================
# 2. SEARCH SETTINGS (Range and Resolution)
# =================================================================
SEARCH_LIMITS = {
    't0': (10412.0, 10412.4, 100),
    'u0': (0.26, 0.325, 200),
    'tau': (53, 61, 200),
    'fbl': (0.29, 0.39, 150)
}


# =================================================================
# 3. THE MATHEMATICAL MODEL (Fitting to Magnitudes)
# =================================================================
def magnitude_model_formula(t, t0, u0, tau, fbl, i_star):
    # Standard Paczynski magnification
    u = np.sqrt(u0 ** 2 + ((t - t0) / tau) ** 2)
    A_u = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))

    # Magnification including blending
    A_obs = A_u * fbl + (1 - fbl)

    # Convert magnification to magnitude
    # Formula: m = m_base - 2.5 * log10(A_obs)
    return i_star - 2.5 * np.log10(A_obs)


# =================================================================
# 4. EXECUTION ENGINE
# =================================================================
def run_grid_analysis():
    try:
        # Assuming columns in Excel are now: x (time), y (magnitude), dy (magnitude error)
        df = pd.read_excel(FILE_PATH)
        t_obs, mag_obs, dmag_obs = df['x'].values, df['y'].values, df['dy'].values
    except Exception as e:
        print(f"Error: {e}");
        return

    t0_v = np.linspace(*SEARCH_LIMITS['t0'])
    u0_v = np.linspace(*SEARCH_LIMITS['u0'])
    tau_v = np.linspace(*SEARCH_LIMITS['tau'])
    fbl_v = np.linspace(*SEARCH_LIMITS['fbl'])

    param_vectors = [t0_v, u0_v, tau_v, fbl_v]
    labels = ['$t_0$', '$u_0$', r'$\tau$', '$f_{bl}$']

    grid_shape = (len(t0_v), len(u0_v), len(tau_v), len(fbl_v))
    total_points = np.prod(grid_shape)
    chisq_grid = np.zeros(grid_shape)

    print(f"Calculating {total_points} grid points for Magnitude Fitting...")
    start_time = time.time()
    last_update = start_time

    for i in range(len(t0_v)):
        for j in range(len(u0_v)):
            for k in range(len(tau_v)):
                for l in range(len(fbl_v)):
                    # Compute model magnitudes
                    mag_model = magnitude_model_formula(t_obs, t0_v[i], u0_v[j], tau_v[k], fbl_v[l], I_STAR)

                    # Chi-squared on Magnitudes
                    chisq_grid[i, j, k, l] = np.sum(((mag_obs - mag_model) / dmag_obs) ** 2)

                    if time.time() - last_update > 15:
                        prog = ((i * len(u0_v) * len(tau_v) * len(fbl_v) + j * len(tau_v) * len(fbl_v) + k * len(
                            fbl_v) + l) / total_points) * 100
                        print(f"Progress: {prog:.1f}% | Elapsed: {int(time.time() - start_time)}s")
                        last_update = time.time()

    min_chi = np.min(chisq_grid)
    idx = np.unravel_index(np.argmin(chisq_grid), grid_shape)
    best_fit = [param_vectors[d][idx[d]] for d in range(4)]

    # --- RESULTS & ERRORS ---
    errors = []
    for d in range(4):
        collapse = tuple(a for a in range(4) if a != d)
        profile = np.min(chisq_grid, axis=collapse)
        mask = profile <= (min_chi + 1.0)
        valid = np.where(mask)[0]
        err = (param_vectors[d][valid.max()] - param_vectors[d][valid.min()]) / 2 if len(valid) > 1 else (
                    param_vectors[d][1] - param_vectors[d][0])
        errors.append(err)

    print(f"\nRESULTS (Min χ²: {min_chi:.4f}):")
    for i in range(4): print(f"{labels[i].strip('$')}: {best_fit[i]:.5f} ± {errors[i]:.5f}")

    # --- PLOTTING ---
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.35, hspace=0.35, right=0.85)

    v_min, v_max = min_chi, min_chi + 5
    levels = np.linspace(v_min, v_max, 50)

    for i in range(4):  # Y-axis
        for j in range(4):  # X-axis
            ax = axes[i, j]
            if i == j:
                collapse = tuple(a for a in range(4) if a != i)
                y_vals = np.min(chisq_grid, axis=collapse)
                ax.plot(param_vectors[i], y_vals, color='blue')
                ax.axhline(min_chi + 1, color='red', linestyle='--')
                ax.set_title(labels[i])
            else:
                axes_to_min = tuple(a for a in range(4) if a != i and a != j)
                z_data = np.min(chisq_grid, axis=axes_to_min)
                z_to_plot = z_data if i < j else z_data.T

                cf = ax.contourf(param_vectors[j], param_vectors[i], z_to_plot, levels=levels, cmap='viridis_r',
                                 extend='max')
                ax.contour(param_vectors[j], param_vectors[i], z_to_plot, levels=[min_chi + 1.0], colors='red',
                           linestyles='dashed', linewidths=1.5)
                ax.scatter(best_fit[j], best_fit[i], color='red', marker='x', s=25)

            if i == 3: ax.set_xlabel(labels[j])
            if j == 0: ax.set_ylabel(labels[i])

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='$\chi^2$ Value')

    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='$\chi^2$ 1D Profile'),
        Line2D([0], [0], color='red', lw=1.5, ls='--', label='1$\sigma$ Interval ($\Delta\chi^2=1$)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markeredgecolor='red', markersize=10,
               label='Best Fit Point')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.45, 0.96), ncol=2, frameon=False)
    plt.suptitle('4D Microlensing Grid Search (Magnitude Fit)', fontsize=16, y=0.99)
    plt.show()


if __name__ == "__main__":
    run_grid_analysis()
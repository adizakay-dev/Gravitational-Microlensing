import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1) FILE PATH & DATA LOADING
# ============================================================
file_path = 'ADD PATH'
df = pd.read_excel(file_path, sheet_name="Sheet1")

x = df["HJD-2460410 [days]"].to_numpy()
y = df["I [mag]"].to_numpy()
sigma = df["dI [mag]"].to_numpy()

# ============================================================
# 2) WEIGHTED PARABOLIC FIT FUNCTION
# ============================================================
def weighted_parabola_fit(x, y, sigma):
    X = np.column_stack([x**2, x, np.ones_like(x)])
    W = np.diag(1.0 / sigma**2)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    a, b, c = beta
    y_fit = a*x**2 + b*x + c
    chi2 = np.sum(((y - y_fit) / sigma)**2)
    dof = len(x) - 3
    x_peak = -b / (2*a)
    y_peak = a*x_peak**2 + b*x_peak + c
    return {"a": a, "b": b, "c": c, "y_fit": y_fit, "chi2": chi2,
            "chi2_red": chi2/dof, "dof": dof, "x_peak": x_peak, "y_peak": y_peak}

real_fit = weighted_parabola_fit(x, y, sigma)

# ============================================================
# 3) PARAMETRIC BOOTSTRAP
# ============================================================
n_boot = 10000
xpeak_boot = np.zeros(n_boot)
ypeak_boot = np.zeros(n_boot)
rng = np.random.default_rng(12345)

for i in range(n_boot):
    y_mock = real_fit["y_fit"] + rng.normal(0, sigma)
    fit_mock = weighted_parabola_fit(x, y_mock, sigma)
    xpeak_boot[i] = fit_mock["x_peak"]
    ypeak_boot[i] = fit_mock["y_peak"]

# ============================================================
# 4) GAUSSIAN FITTING UTILITY
# ============================================================
def gaussian_model(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def fit_gaussian_to_hist(data, bins=40):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Guesses: Max height, Mean of data, Std of data
    p0 = [np.max(counts), np.mean(data), np.std(data)]
    popt, _ = curve_fit(gaussian_model, bin_centers, counts, p0=p0)
    return popt, bin_edges

# ============================================================
# 5) PLOTTING AND RESULTS
# ============================================================

# (a) Peak Time Distribution
popt_x, edges_x = fit_gaussian_to_hist(xpeak_boot)
amp_x, mu_x, std_x = popt_x

plt.figure(figsize=(8, 5))
plt.hist(xpeak_boot, bins=40, alpha=0.7, color='C0', label='Bootstrap Data')
x_plot = np.linspace(edges_x[0], edges_x[-1], 200)
plt.plot(x_plot, gaussian_model(x_plot, *popt_x), 'r-', lw=2,
         label=f'Gaussian Fit\n$\mu$: {mu_x:.2f}\n$\sigma$: {std_x:.2f}')
#plt.axvline(real_fit["x_peak"], color='black', linestyle=':', label='Observed $t_0$')
plt.title("Bootstrap Distribution of Peak Time ($t_0$)")
plt.xlabel("t_0 (HJD-2460410 [days])")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# (b) Peak Magnitude Distribution
popt_I, edges_I = fit_gaussian_to_hist(ypeak_boot)
amp_I, mu_I, std_I = popt_I

plt.figure(figsize=(8, 5))
plt.hist(ypeak_boot, bins=40, alpha=0.7, color='C0', label='Bootstrap Data')
I_plot = np.linspace(edges_I[0], edges_I[-1], 200)
plt.plot(I_plot, gaussian_model(I_plot, *popt_I), 'r-', lw=2,
         label=f'Gaussian Fit\n$\mu$: {mu_I:.4f}\n$\sigma$: {std_I:.4f}')
#plt.axvline(real_fit["y_peak"], color='black', linestyle=':', label='Observed $I_{peak}$')
plt.title("Bootstrap Distribution of Peak Magnitude")
plt.xlabel("I_peak [mag]")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Final Console Output
print("--- Gaussian Fit Results ---")
print(f"PEAK TIME (x_peak):")
print(f"  Amplitude = {amp_x:.2f}")
print(f"  Mean (mu) = {mu_x:.6f}")
print(f"  Std Dev (sigma) = {std_x:.6f}")
print(f"\nPEAK MAGNITUDE (I_peak):")
print(f"  Amplitude = {amp_I:.2f}")
print(f"  Mean (mu) = {mu_I:.6f}")
print(f"  Std Dev (sigma) = {std_I:.6f}")
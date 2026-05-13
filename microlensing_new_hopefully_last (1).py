import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ==========================================================
# STEP 1: SETUP DATA AND CONSTANTS
# ==========================================================
my_file_path = "ADD PATH"
coords = "17:13:07.34 -32:55:17.6"

data = mm.MulensData(file_name=my_file_path, delimiter=",", add_2450000=False, skiprows=1)

# Fixed baseline parameters
t_0 = 2460412.17
u_0 = 0.300
t_E = 55.978
t_0_par = 2460412.0


# ==========================================================
# STEP 2: DEFINE OPTIMIZATION FUNCTION
# ==========================================================
def chi2_func(pi_values, data, coords):
    """Function to minimize: returns chi2 for given parallax pi_N and pi_E."""
    pi_E_N, pi_E_E = pi_values

    # Create a model with the current parallax guesses
    params = {
        't_0': t_0, 'u_0': u_0, 't_E': t_E,
        'pi_E_N': pi_E_N, 'pi_E_E': pi_E_E,
        't_0_par': t_0_par
    }

    model = mm.Model(params, coords=coords)
    event = mm.Event(datasets=data, model=model)

    # We must fit fluxes (source/blend) to get an accurate chi2
    event.fit_fluxes()
    return event.get_chi2()


# ==========================================================
# STEP 3: RUN THE MINIMIZER
# ==========================================================
# Initial guess for [pi_E_N, pi_E_E]
initial_guess = [0.1, -0.1]

print("Optimizing parallax parameters... this may take a moment.")
result = minimize(chi2_func, initial_guess, args=(data, coords), method='Nelder-Mead')

best_pi_N, best_pi_E = result.x
best_chi2_px = result.fun

# ==========================================================
# STEP 4: CALCULATE STANDARD MODEL (NO PARALLAX)
# ==========================================================
model_std = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E}, coords=coords)
event_std = mm.Event(datasets=data, model=model_std)
event_std.fit_fluxes()
chi2_std = event_std.get_chi2()

# ==========================================================
# STEP 5: FINAL COMPARISON AND PLOTTING
# ==========================================================
print("-" * 30)
print(f"Standard Model Chi2: {chi2_std:.2f}")
print(f"Best Parallax Chi2:  {best_chi2_px:.2f}")
print(f"Best pi_E_N: {best_pi_N:.4f}")
print(f"Best pi_E_E: {best_pi_E:.4f}")
print("-" * 30)

# Create the best parallax model for plotting
best_params_px = {
    't_0': t_0, 'u_0': u_0, 't_E': t_E,
    'pi_E_N': best_pi_N, 'pi_E_E': best_pi_E,
    't_0_par': t_0_par
}
event_px_best = mm.Event(datasets=data, model=mm.Model(best_params_px, coords=coords))

plt.figure(figsize=(12, 6))
event_std.plot_data(color='black', s=5, alpha=0.5, label='Data')
event_std.plot_model(color='blue', linestyle='--', label=f'Standard (Chi2: {chi2_std:.1f})')
event_px_best.plot_model(color='red', label=f'Best Parallax (Chi2: {best_chi2_px:.1f})')

plt.title(f"Parallax Optimization: pi_N={best_pi_N:.3f}, pi_E={best_pi_E:.3f}")
plt.xlabel("Time HJD [days]")
plt.ylabel("Magnitude")
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
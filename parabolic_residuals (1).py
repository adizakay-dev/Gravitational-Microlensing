import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt


def parabola(x, a, b, c):
    """Standard quadratic form: y = ax^2 + bx + c"""
    return a * x ** 2 + b * x + c


def fit_light_curve(file_path, initial_peak_points=4):
    # 1. Load the data
    # Works for both CSV and Excel on Mac/Windows
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Assumes columns are [time, mag, error]
    df.columns = ['x', 'y', 'y_err']
    df = df.sort_values(by='x').reset_index(drop=True)

    # 2. Find the peak (lowest magnitude)
    peak_idx = df['y'].idxmin()

    # 3. Initialize the fitting region
    # We take the N points closest to the peak index
    half_n = initial_peak_points // 2
    start_idx = max(0, peak_idx - half_n)
    end_idx = min(len(df), start_idx + initial_peak_points)

    results = []
    step = 1

    print(f"{'Step':<6} | {'n_points':<8} | {'a (curvature)':<15} | {'b':<15} | {'c (peak mag)':<15}")
    print("-" * 75)

    # 4. Iterative Fitting
    # 4. Iterative Fitting and Plotting

    # 4. Iterative Fitting from 3 to 10 points
    for n_points in range(6, 7):
        # Center the window on the peak
        half_n = n_points // 2
        start_idx = max(0, peak_idx - half_n)
        end_idx = min(len(df), start_idx + n_points)

        current_data = df.iloc[start_idx:end_idx]

        try:
            popt, pcov = curve_fit(parabola, current_data['x'], current_data['y'],
                                   sigma=current_data['y_err'], absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            a, b, c = popt

            # Create a figure with two subplots: [Top: Light Curve, Bottom: Residuals]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})

            # --- TOP PLOT: Light Curve ---
            ax1.scatter(df['x'], df['y'], color='lightgrey', alpha=0.5, label='Full Dataset')
            ax1.errorbar(current_data['x'], current_data['y'], yerr=current_data['y_err'],
                         fmt='o', color='black', label=f'Fitted Points (n={n_points})')

            x_range = np.linspace(df['x'].min(), df['x'].max(), 500)
            ax1.plot(x_range, parabola(x_range, *popt), 'r--', alpha=0.7, label='Extended Fit')
            ax1.plot(np.linspace(current_data['x'].min(), current_data['x'].max(), 100),
                     parabola(np.linspace(current_data['x'].min(), current_data['x'].max(), 100), *popt),
                     'r-', lw=3, label='Local Fit')

            ax1.invert_yaxis()
            ax1.set_ylabel('Magnitude')
            ax1.set_title(f'Parabolic Fit for n={n_points} points')
            ax1.legend()

            # --- BOTTOM PLOT: Residuals ---
            # Calculate residuals for the ENTIRE dataset based on this specific fit
            residuals = df['y'] - parabola(df['x'], *popt)

            ax2.errorbar(df['x'], residuals, yerr=df['y_err'], fmt='o', color='blue', alpha=0.4)
            ax2.axhline(0, color='black', linestyle='--')  # Zero line
            ax2.set_ylabel('Residuals')
            ax2.set_xlabel('Time HJD-2460383 [days]')

            # Add fit parameters text to the top plot
            stats_text = (f'a = {a:.4f}±{perr[0]:.4f}\nb = {b:.4f}±{perr[1]:.4f}\nc = {c:.4f}±{perr[2]:.4f}')
            ax1.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                         bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            plt.tight_layout()
            plt.show()

            print(f"Points: {n_points:<2} | a: {a:1.4f}±{perr[0]:.4f} | b: {b:1.4f}±{perr[1]:.4f}")

        except Exception as e:
            print(f"Fit failed for n={n_points}: {e}")
            continue

    # while start_idx >= 0 and end_idx <= len(df):
    #     current_data = df.iloc[start_idx:end_idx]
    #
    #     if len(current_data) < 3:
    #         break
    #
    #     try:
    #         popt, pcov = curve_fit(parabola, current_data['x'], current_data['y'],
    #                                sigma=current_data['y_err'], absolute_sigma=True)
    #         perr = np.sqrt(np.diag(pcov))
    #         a, b, c = popt
    #
    #         # --- Plotting Section ---
    #         plt.figure(figsize=(10, 6))
    #
    #         # Plot all data in the background
    #         plt.scatter(df['x'], df['y'], color='lightgrey', alpha=0.5, label='All Data')
    #
    #         # Plot current subset with error bars
    #         plt.errorbar(current_data['x'], current_data['y'], yerr=current_data['y_err'],
    #                      fmt='o', color='black', label=f'Fit Points (n={len(current_data)})')
    #
    #         # Generate smooth x-values for the parabola line
    #         x_fit = np.linspace(current_data['x'].min(), current_data['x'].max(), 100)
    #         y_fit = parabola(x_fit, *popt)
    #         plt.plot(x_fit, y_fit, 'r-', lw=2, label='Parabolic Fit')
    #
    #         # Formatting the graph
    #         plt.gca().invert_yaxis()  # Magnitudes are inverted
    #         plt.xlabel('Time')
    #         plt.ylabel('Magnitude')
    #         plt.title(f'Step {step}: Iterative Parabolic Fit')
    #
    #         # Display parameters on the graph
    #         stats_text = (f'a = {a:.4f} ± {perr[0]:.4f}\n'
    #                       f'b = {b:.4f} ± {perr[1]:.4f}\n'
    #                       f'c = {c:.4f} ± {perr[2]:.4f}')
    #
    #         plt.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
    #                      bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    #
    #         plt.legend()
    #         plt.grid(True, linestyle=':', alpha=0.6)
    #         plt.show()  # This will pause the code until you close the window
    #         # ------------------------
    #
    #         results.append({'step': step, 'params': popt, 'errors': perr})
    #         print(
    #             f"{step:<6} | {len(current_data):<8} | {a:1.4f}±{perr[0]:.4f} | {b:1.4f}±{perr[1]:.4f} | {c:1.4f}±{perr[2]:.4f}")
    #
    #     except Exception as e:
    #         print(f"Fit failed at step {step}: {e}")
    #         break
    #
    #     start_idx -= 1
    #     end_idx += 1
    #     step += 1


# --- Configuration ---
# Mac path example: /Users/yourname/Desktop/data.csv
file_location = 'ADD PATH'
points_to_start = 21

if __name__ == "__main__":
    if os.path.exists(file_location):
        final_results = fit_light_curve(file_location, points_to_start)
    else:
        print(f"File not found at {file_location}. Please check the path.")
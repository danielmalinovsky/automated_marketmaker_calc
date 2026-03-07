# VolumePathAnalysis - Research Visualization Package

## Overview

`VolumePathAnalysis` is a specialized visualization class designed for research paper presentations of path-independent volume analysis in Constant Product AMMs. It combines theoretical volume bound calculations with empirical simulation data to validate the mathematical model.

## Quick Start

### Installation

The class is located in the `automated_marketmaker_calc` package and can be imported directly:

```python
from automated_marketmaker_calc import VolumePathAnalysis
```

### Basic Usage

```python
import numpy as np
from automated_marketmaker_calc import VolumePathAnalysis

# Initialize with pool parameters
x0 = 1000  # Initial ETH reserves
P0 = 4000  # Initial price (USDC/ETH)

analyzer = VolumePathAnalysis(x0=x0, P0=P0)

# Create visualization with simulation overlay
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
    price_ratio_range=(0.25, 4.0),
    color_by='error_pct'
)
plt.show()

# Print validation statistics
analyzer.print_summary(simulation_results)
```

## Integration with calibrated_prices.ipynb

### Workflow

1. **Run Payoff Pipeline**
   ```python
   from automated_marketmaker_calc import Payoff
   
   payoff = Payoff()
   payoff.pipeline(
       pool_fee=0.01,
       amount_x_pool_t0=1000,
       amount_y_pool_t0=10000,
       total_investment_x=100,
       FX_timeseries=simulated_prices,
       volume_timeseries=simulated_volumes,
       max_paths=500,
       fee_split=0.5,
       use_volume_decomposition=False,
       drop_insufficient_volume=False
   )
   ```

2. **Extract Simulation Results from paths_df**
   ```python
   # Convert payoff.paths_df to simulation_results format
   simulation_results = []
   
   for idx, row in payoff.paths_df.iterrows():
       P0 = row['FX_t0']
       P1 = row['FX_tN']
       V_actual = row['V_total_vol']
       
       # Calculate theoretical bound
       x0_val = row['amount_x_pool_t0']
       V_bound = x0_val * abs(1 - np.sqrt(P0 / P1))
       
       simulation_results.append({
           'P0': P0,
           'P1': P1,
           'price_change_pct': (P1 - P0) / P0 * 100,
           'V_bound': V_bound,
           'V_actual': V_actual,
           'error': V_bound - V_actual,
           'error_pct': ((V_bound - V_actual) / V_actual * 100) if V_actual != 0 else 0,
       })
   ```

3. **Create Research Visualization**
   ```python
   from automated_marketmaker_calc import VolumePathAnalysis
   
   analyzer = VolumePathAnalysis(x0=x0_val, P0=P0_initial)
   
   # Main visualization: Theory vs Simulation
   fig1, ax1 = analyzer.plot_theoretical_with_simulation_overlay(
       simulation_results=simulation_results,
       price_ratio_range=(0.5, 2.0),
       figsize=(12, 7),
       color_by='error_pct',
       alpha_scatter=0.5,
       scatter_size=40,
       save_path='volume_theory_vs_sim.png'
   )
   
   # Error analysis
   fig2, ax2 = analyzer.plot_error_distribution(
       simulation_results=simulation_results,
       save_path='volume_error_dist.png'
   )
   
   # Accuracy robustness
   fig3, ax3 = analyzer.plot_accuracy_vs_price_movement(
       simulation_results=simulation_results,
       save_path='volume_accuracy_vs_movement.png'
   )
   ```

4. **Validate Results**
   ```python
   analyzer.print_summary(simulation_results)
   ```

## Available Methods

### `__init__(x0: float, P0: float)`
Initialize the analyzer with pool parameters.

**Parameters:**
- `x0` (float): Initial reserves of token X (denominator token, e.g., ETH)
- `P0` (float): Initial price (Y per X, e.g., USDC per ETH)

---

### `calculate_theoretical_bound(price_ratios: np.ndarray) -> np.ndarray`
Calculate theoretical volume bound for given price ratios.

**Formula:** $V_{bound} = x_0 \cdot |1 - \sqrt{P_0 / P_1}|$

**Parameters:**
- `price_ratios` (np.ndarray): Array of price ratios (P₁/P₀)

**Returns:**
- np.ndarray: Volume bounds for each price ratio

---

### `plot_theoretical_with_simulation_overlay(...) -> Tuple[plt.Figure, plt.Axes]`
**Main visualization** - Plot theoretical volume bound line with simulation scatter overlay.

**Parameters:**
- `simulation_results` (list): List of simulation dictionaries
- `price_ratio_range` (Tuple): Min/max price ratio for x-axis (default: (0.25, 4.0))
- `figsize` (Tuple): Figure size in inches (default: (12, 7))
- `color_by` (str): Coloring strategy:
  - `'error_pct'`: Color by estimation error (default) - Red=high error, Green=low error
  - `'price_change'`: Color by absolute price change magnitude - Plasma colormap
  - `'uniform'`: Single green color for all points
- `alpha_scatter` (float): Transparency of scatter points 0-1 (default: 0.5)
- `scatter_size` (int): Size of scatter markers (default: 40)
- `save_path` (Optional[str]): Path to save figure (default: None = no save)

**Returns:**
- Tuple[plt.Figure, plt.Axes]: Figure and axes for further customization

**Example:**
```python
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=results,
    color_by='error_pct',
    save_path='fig1_theory_vs_sim.png'
)
```

---

### `plot_error_distribution(...) -> Tuple[plt.Figure, plt.Axes]`
Plot histogram of errors between theoretical bound and simulation results.

**Parameters:**
- `simulation_results` (list): List of simulation dictionaries
- `figsize` (Tuple): Figure size (default: (10, 6))
- `bins` (int): Number of histogram bins (default: 30)
- `save_path` (Optional[str]): Path to save figure

**Returns:**
- Tuple[plt.Figure, plt.Axes]: Figure and axes objects

---

### `plot_accuracy_vs_price_movement(...) -> Tuple[plt.Figure, plt.Axes]`
Plot percentage error vs price change to show model robustness across price ranges.

**Parameters:**
- `simulation_results` (list): List of simulation dictionaries
- `figsize` (Tuple): Figure size (default: (10, 6))
- `save_path` (Optional[str]): Path to save figure

**Returns:**
- Tuple[plt.Figure, plt.Axes]: Figure and axes objects

---

### `summary_statistics(simulation_results: list) -> dict`
Calculate comprehensive accuracy metrics.

**Returns dict with:**
- `num_simulations`: Number of simulations
- `mean_absolute_error_eth`: Mean error in ETH
- `std_error_eth`: Standard deviation of errors
- `mean_absolute_error_pct`: Mean percentage error
- `median_absolute_error_pct`: Median percentage error
- `correlation`: Correlation between V_bound and V_actual
- `accuracy_within_1pct`: % of simulations within ±1% error
- `accuracy_within_5pct`: % of simulations within ±5% error
- `accuracy_within_10pct`: % of simulations within ±10% error
- `percentile_95_error_pct`: 95th percentile error

---

### `print_summary(simulation_results: list) -> None`
Print formatted summary statistics to console.

## Input Format: Simulation Results

The `simulation_results` parameter expects a list of dictionaries with the following structure:

```python
simulation_results = [
    {
        'P0': float,              # Initial price
        'P1': float,              # Final price
        'price_change_pct': float, # Percentage change: (P1 - P0) / P0 * 100
        'V_bound': float,         # Theoretical volume requirement
        'V_actual': float,        # Actual volume from simulation
        'error': float,           # V_bound - V_actual (in absolute units)
        'error_pct': float,       # ((V_bound - V_actual) / V_actual) * 100
    },
    # ... more results ...
]
```

## Visualization Features

### Color Coding Strategies

#### 1. By Error Percentage (Default)
```python
color_by='error_pct'  # Red (high error) → Green (low error)
```
- Shows where the theoretical model underestimates or overestimates actual volumes
- Helps identify if errors correlate with specific price ranges
- Useful for paper: demonstrates model accuracy visually

#### 2. By Price Change
```python
color_by='price_change'  # Plasma colormap by |ΔP|
```
- Shows if model accuracy depends on magnitude of price movement
- Larger colors indicate larger price moves
- Useful for paper: demonstrates robustness across market conditions

#### 3. Uniform Green
```python
color_by='uniform'  # All points same color
```
- Clean, publication-ready appearance
- No information in color encoding
- Useful for paper: focuses on scatter distribution only

### Styling Features

- **Theoretical Line**: Blue, linewidth=3 (clearly visible, main reference)
- **Initial Price Reference**: Red dashed line at P/P₀ = 1
- **Scatter Points**: Semi-transparent (α=0.5 default) with edge colors
- **Annotations**: Yellow boxes for key price points (0.5x, 1.0x, 4.0x moves)
- **Grid**: Light transparency (α=0.3) for readability
- **Legend**: Positioned to avoid data overlap

## Example Outputs

### Typical Summary Statistics
```
V_BOUND ACCURACY ANALYSIS - SUMMARY STATISTICS
================================================================
Total Simulations: 500

Error Metrics:
  Mean Absolute Error: 0.021543 ETH
  Std Dev Error: 0.043212 ETH
  Mean % Error: 0.2847%
  Median % Error: 0.1923%
  95th Percentile Error: 1.2134%

Correlation: 0.998764

Accuracy Tolerance Bands:
  Within ±1%:  92.4%
  Within ±5%:  99.2%
  Within ±10%: 100.0%

KEY INSIGHT:
High correlation and tight error bounds confirm V_bound formula
accurately predicts volume required for price movements.
================================================================
```

## For Research Papers

### Recommended Figure Set

**Figure 1: Main Validation Plot**
```python
fig1, ax1 = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=results,
    price_ratio_range=(0.25, 4.0),
    color_by='error_pct',
    save_path='fig1_main_validation.png'
)
```
Caption: "Empirical simulations (colored by estimation error) validate the path-independent volume bound formula across price ranges."

**Figure 2: Error Distribution**
```python
fig2, ax2 = analyzer.plot_error_distribution(
    simulation_results=results,
    save_path='fig2_error_distribution.png'
)
```
Caption: "Distribution of errors shows near-zero mean error, indicating V_bound is unbiased estimator of actual volume."

**Figure 3: Robustness**
```python
fig3, ax3 = analyzer.plot_accuracy_vs_price_movement(
    simulation_results=results,
    save_path='fig3_robustness.png'
)
```
Caption: "Model accuracy remains consistent across all price movement magnitudes, confirming formula applicability."

## Troubleshooting

### Empty Simulation Results
```python
# Error: ValueError: simulation_results cannot be empty
# Solution: Ensure simulations completed successfully
print(len(simulation_results))  # Should be > 0
```

### NaN in Error Calculation
```python
# Can occur if V_actual = 0 for some paths
# Already handled in error_pct calculation as conditional check
# If needed, filter: [r for r in results if r['V_actual'] > 0.001]
```

### Figure Not Saving
```python
# Ensure directory exists and has write permissions
import os
save_dir = 'figures/'
os.makedirs(save_dir, exist_ok=True)

fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=results,
    save_path='figures/my_figure.png'
)
```

## Citation

If using this visualization class in research, cite:
```
VolumePathAnalysis (2026). Automated Market Maker Calculation Package.
https://github.com/your-repo/automated_marketmaker_calc
```

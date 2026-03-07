# Quick Integration Guide for calibrated_prices.ipynb

## Step-by-Step: Adding Research Visualizations

### Step 1: Import the Visualization Class
Add this cell at the top of calibrated_prices.ipynb:

```python
from automated_marketmaker_calc import VolumePathAnalysis
import numpy as np
```

---

### Step 2: After Running Payoff Pipeline
After your `payoff.pipeline()` call, extract the simulation data:

```python
# Get parameters from payoff results
amount_x_pool_t0 = payoff.paths_df['amount_x_pool_t0'].iloc[0]
amount_y_pool_t0 = payoff.paths_df['amount_y_pool_t0'].iloc[0]

# Calculate initial price
P0_initial = amount_y_pool_t0 / amount_x_pool_t0

# Convert paths_df to simulation_results format
simulation_results = []

for idx, row in payoff.paths_df.iterrows():
    try:
        P0 = row['FX_t0']
        P1 = row['FX_tN']
        V_actual = row['V_total_vol']
        
        # Avoid division by zero
        if P1 == 0 or V_actual < 0.001:
            continue
        
        # Calculate theoretical volume bound
        x0_val = amount_x_pool_t0
        V_bound = x0_val * abs(1 - np.sqrt(P0 / P1))
        
        error = V_bound - V_actual
        error_pct = (error / V_actual * 100) if V_actual != 0 else 0
        
        simulation_results.append({
            'P0': P0,
            'P1': P1,
            'price_change_pct': (P1 - P0) / P0 * 100,
            'V_bound': V_bound,
            'V_actual': V_actual,
            'error': error,
            'error_pct': error_pct,
        })
    except Exception as e:
        print(f"Skipped row {idx}: {e}")
        continue

print(f"Extracted {len(simulation_results)} valid simulation results")
```

---

### Step 3: Create the Analyzer Instance
```python
# Initialize analyzer with pool parameters
analyzer = VolumePathAnalysis(x0=amount_x_pool_t0, P0=P0_initial)
```

---

### Step 4: Generate Main Research Visualization
```python
# Create the primary figure: Theory vs Simulation overlay
fig1, ax1 = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
    price_ratio_range=(0.5, 2.0),  # Adjust based on your price ranges
    figsize=(12, 7),
    color_by='error_pct',  # Use 'error_pct', 'price_change', or 'uniform'
    alpha_scatter=0.5,
    scatter_size=40,
    save_path=None  # Set to filename to save: 'fig_volume_validation.png'
)
plt.show()
```

---

### Step 5: Generate Supporting Visualizations
```python
# Error distribution analysis
fig2, ax2 = analyzer.plot_error_distribution(
    simulation_results=simulation_results,
    figsize=(10, 6),
    bins=30,
    save_path=None  # Set to 'fig_error_dist.png' to save
)
plt.show()

# Model robustness across price movements
fig3, ax3 = analyzer.plot_accuracy_vs_price_movement(
    simulation_results=simulation_results,
    figsize=(10, 6),
    save_path=None  # Set to 'fig_robustness.png' to save
)
plt.show()
```

---

### Step 6: Validation Statistics
```python
# Print comprehensive summary
analyzer.print_summary(simulation_results)

# Or get dict of stats for further analysis
stats = analyzer.summary_statistics(simulation_results)
print(f"Correlation: {stats['correlation']:.6f}")
print(f"Mean Error: {stats['mean_absolute_error_pct']:.4f}%")
print(f"Accuracy within ±1%: {stats['accuracy_within_1pct']:.1f}%")
```

---

## Color-Coding Options Explained

### Option 1: By Estimation Error (Recommended)
```python
color_by='error_pct'
```
- **Green points**: Simulation matches theory well (low error)
- **Red points**: Simulation deviates from theory (high error)
- **Best for**: Demonstrating model accuracy visually
- **Red-Yellow-Green (RdYlGn_r) colormap** for publication quality

### Option 2: By Price Change Magnitude
```python
color_by='price_change'
```
- **Dark points**: Small price movements
- **Bright points**: Large price movements
- **Best for**: Showing consistency across market conditions
- **Plasma colormap** for visual appeal

### Option 3: Uniform Color
```python
color_by='uniform'
```
- **All green points**: Clean, publication-ready
- **No color information**: Focuses on distribution only
- **Best for**: Simple presentations

---

## Typical Output Example

```
V_BOUND ACCURACY ANALYSIS - SUMMARY STATISTICS
================================================================
Total Simulations: 432

Error Metrics:
  Mean Absolute Error: 0.015432 ETH
  Std Dev Error: 0.038231 ETH
  Mean % Error: 0.3421%
  Median % Error: 0.1876%
  95th Percentile Error: 1.4567%

Correlation: 0.998921

Accuracy Tolerance Bands:
  Within ±1%:  93.5%
  Within ±5%:  99.3%
  Within ±10%: 100.0%

KEY INSIGHT:
High correlation and tight error bounds confirm V_bound formula
accurately predicts volume required for price movements.
================================================================
```

---

## Common Issues & Solutions

### Issue: Extracted 0 valid results
```python
# Check for NaN or invalid prices
print(payoff.paths_df[['FX_t0', 'FX_tN', 'V_total_vol']].head())
print(payoff.paths_df.isnull().sum())

# Filter out invalid rows before extraction
payoff.paths_df = payoff.paths_df.dropna()
```

### Issue: Error calculation gives NaN
```python
# Likely V_actual = 0 for some paths
# Solution: Already handled - use conditional check:
error_pct = (error / V_actual * 100) if V_actual > 0 else 0
```

### Issue: Scatter points overlapped, hard to see
```python
# Adjust transparency or size
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
    alpha_scatter=0.3,  # More transparent
    scatter_size=30,    # Smaller points
)
```

### Issue: Need different price ratio range
```python
# Adapt to your actual price movements
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
    price_ratio_range=(0.8, 1.5),  # For small moves
    # OR
    # price_ratio_range=(0.25, 4.0),  # For large moves
)
```

---

## Saving Figures for Paper

### Save Individual Figures
```python
fig1, ax1 = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
    save_path='figures/fig1_volume_validation.png'
)

fig2, ax2 = analyzer.plot_error_distribution(
    simulation_results=simulation_results,
    save_path='figures/fig2_error_distribution.png'
)

fig3, ax3 = analyzer.plot_accuracy_vs_price_movement(
    simulation_results=simulation_results,
    save_path='figures/fig3_robustness.png'
)
```

### Create All Figures at Once
```python
import os

# Ensure directory exists
os.makedirs('paper_figures', exist_ok=True)

# Generate all figures with saving
fig1, _ = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results, save_path='paper_figures/01_theory_vs_sim.png')
fig2, _ = analyzer.plot_error_distribution(
    simulation_results, save_path='paper_figures/02_error_dist.png')
fig3, _ = analyzer.plot_accuracy_vs_price_movement(
    simulation_results, save_path='paper_figures/03_robustness.png')

print("All figures saved to paper_figures/")
```

---

## Customizing Figures Further

### Modify Title and Labels
```python
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=simulation_results,
)

# Customize after creation
ax.set_title('Custom Title: Path-Independent Volume Requirements\nwith Empirical Validation', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Price Ratio (P₁/P₀)', fontsize=13)
ax.set_ylabel('Volume Required (ETH)', fontsize=13)

plt.tight_layout()
plt.savefig('custom_figure.png', dpi=300, bbox_inches='tight')
```

### Combine Multiple Plots
```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: Theory vs Simulation (spans 2 rows)
ax1 = fig.add_subplot(gs[:, 0])
# ... customize ax1 ...

# Plot 2: Error distribution (top right)
ax2 = fig.add_subplot(gs[0, 1])
# ... customize ax2 ...

# Plot 3: Robustness (bottom right)
ax3 = fig.add_subplot(gs[1, 1])
# ... customize ax3 ...

plt.savefig('combined_figure.png', dpi=300, bbox_inches='tight')
```

---

## Next Steps

1. ✓ Import `VolumePathAnalysis` in calibrated_prices.ipynb
2. ✓ Extract simulation data from `payoff.paths_df`
3. ✓ Create analyzer instance
4. ✓ Generate visualizations
5. ✓ Validate with summary statistics
6. ✓ Save figures for paper/presentation
7. Consider: Customize colors, styles, or combine plots for publication

Detailed documentation available in: `RESEARCH_VISUALIZATIONS_GUIDE.md`

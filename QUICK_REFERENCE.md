# VolumePathAnalysis - Quick Reference Card

## One-Page Cheat Sheet

### Import
```python
from automated_marketmaker_calc import VolumePathAnalysis
```

### Initialize
```python
analyzer = VolumePathAnalysis(x0=1000, P0=4000)
```

### Main Visualization (Theory + Simulation)
```python
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results=results,
    price_ratio_range=(0.25, 4.0),
    figsize=(12, 7),
    color_by='error_pct',           # or 'price_change', 'uniform'
    alpha_scatter=0.5,
    scatter_size=40,
    save_path='fig1.png'            # optional
)
```

### Error Distribution
```python
fig, ax = analyzer.plot_error_distribution(
    simulation_results=results,
    figsize=(10, 6),
    bins=30,
    save_path='fig2.png'
)
```

### Robustness Check
```python
fig, ax = analyzer.plot_accuracy_vs_price_movement(
    simulation_results=results,
    figsize=(10, 6),
    save_path='fig3.png'
)
```

### Statistics
```python
# Formatted output
analyzer.print_summary(results)

# Or get dictionary
stats = analyzer.summary_statistics(results)
print(f"Correlation: {stats['correlation']:.6f}")
print(f"Mean Error: {stats['mean_absolute_error_pct']:.4f}%")
```

---

## Input Format

```python
simulation_results = [
    {
        'P0': 4000,                    # Initial price
        'P1': 3200,                    # Final price
        'price_change_pct': -20,       # Percentage change
        'V_bound': 192.5,              # Theory
        'V_actual': 191.3,             # Empirical
        'error': 1.2,                  # Difference
        'error_pct': 0.628,            # Relative error %
    },
    # ... more ...
]
```

---

## Color-Coding Options

| Option | Meaning | Colormap | Best For |
|--------|---------|----------|----------|
| `'error_pct'` | Red=high error, Green=good | RdYlGn_r | Accuracy visualization |
| `'price_change'` | Dark=small move, Bright=big | Plasma | Robustness check |
| `'uniform'` | All green | - | Clean presentation |

---

## Output Statistics Explained

| Metric | What It Means | Target |
|--------|--------------|--------|
| Correlation | How well theory matches sim | > 0.99 |
| Mean % Error | Average deviation | < 1% |
| ±1% Accuracy | % of points within ±1% | > 80% |
| ±5% Accuracy | % of points within ±5% | > 95% |

---

## Common Workflows

### For Research Paper
```python
# Generate all figures for publication
figs = {}
figs['theory'] = analyzer.plot_theoretical_with_simulation_overlay(
    results, color_by='error_pct', save_path='fig1_theory_vs_sim.png')
figs['errors'] = analyzer.plot_error_distribution(
    results, save_path='fig2_errors.png')
figs['robust'] = analyzer.plot_accuracy_vs_price_movement(
    results, save_path='fig3_robustness.png')

analyzer.print_summary(results)  # For paper text
```

### For Presentation
```python
# Quick summary check
analyzer.plot_theoretical_with_simulation_overlay(
    results, 
    color_by='uniform',  # Clean look
    alpha_scatter=0.7,
    scatter_size=50
)
analyzer.print_summary(results)
```

### For Debugging
```python
# Check if data is good
print(len(results), "simulations")
print(f"Price range: {min(r['price_change_pct'] for r in results):.1f}% to "
      f"{max(r['price_change_pct'] for r in results):.1f}%")

stats = analyzer.summary_statistics(results)
if stats['correlation'] < 0.95:
    print("⚠️  Low correlation - check data quality")
```

---

## From calibrated_prices.ipynb

```python
# After payoff.pipeline():

# Convert paths_df to simulation_results
simulation_results = []
for idx, row in payoff.paths_df.iterrows():
    P0, P1 = row['FX_t0'], row['FX_tN']
    V_actual = row['V_total_vol']
    x0_val = row['amount_x_pool_t0']
    
    V_bound = x0_val * abs(1 - np.sqrt(P0/P1))
    
    simulation_results.append({
        'P0': P0, 'P1': P1,
        'price_change_pct': (P1 - P0) / P0 * 100,
        'V_bound': V_bound,
        'V_actual': V_actual,
        'error': V_bound - V_actual,
        'error_pct': (V_bound - V_actual) / V_actual * 100,
    })

# Visualize
analyzer = VolumePathAnalysis(x0=x0_val, P0=P0)
fig, ax = analyzer.plot_theoretical_with_simulation_overlay(
    simulation_results, color_by='error_pct'
)
analyzer.print_summary(simulation_results)
```

---

## Troubleshooting Cheat Sheet

| Problem | Solution |
|---------|----------|
| Empty results | Check `len(simulation_results)` > 0 |
| NaN in plot | Filter V_actual > 0.001 before passing |
| Points hard to see | Increase `alpha_scatter`, decrease `scatter_size` |
| Wrong price range | Adjust `price_ratio_range` parameter |
| Bad error values | Ensure error_pct = (V_bound - V_actual) / V_actual * 100 |
| Can't import | Run `pip install -e .` in package directory |

---

## Key Statistics Formulas

```
V_bound = x0 * |1 - √(P0/P1)|

error = V_bound - V_actual

error_pct = (error / V_actual) * 100

correlation = corrcoef(V_bound, V_actual)[0,1]

accuracy_within_pct = sum(|error_pct| <= pct) / len(results) * 100
```

---

## File Locations

```
Main class:     automated_marketmaker_calc/research_visualizations.py
Import from:    automated_marketmaker_calc/__init__.py
Full guide:     RESEARCH_VISUALIZATIONS_GUIDE.md
Integration:    CALIBRATED_PRICES_INTEGRATION.md
Architecture:   RESEARCH_VISUALIZATIONS_ARCHITECTURE.md
Examples:       volume_bounds.ipynb (last 4 cells)
```

---

## Method Signatures Quick Ref

```python
__init__(x0: float, P0: float) → VolumePathAnalysis

calculate_theoretical_bound(price_ratios: np.ndarray) 
    → np.ndarray

extract_simulation_data(simulation_results: list) 
    → Tuple[np.ndarray, np.ndarray, np.ndarray]

plot_theoretical_with_simulation_overlay(
    simulation_results, price_ratio_range=(0.25, 4.0),
    figsize=(12, 7), color_by='error_pct',
    alpha_scatter=0.5, scatter_size=40, save_path=None
) → Tuple[Figure, Axes]

plot_error_distribution(
    simulation_results, figsize=(10, 6), bins=30, save_path=None
) → Tuple[Figure, Axes]

plot_accuracy_vs_price_movement(
    simulation_results, figsize=(10, 6), save_path=None
) → Tuple[Figure, Axes]

summary_statistics(simulation_results: list) → dict

print_summary(simulation_results: list) → None
```

---

## Plot Output Summary

| Plot | Type | Shows | Use For |
|------|------|-------|---------|
| `plot_theoretical_...` | Line + Scatter | Theory vs data | Main figure |
| `plot_error_distribution` | Histogram | Error spread | Statistical proof |
| `plot_accuracy_...` | Scatter | Error vs movement | Robustness |

---

## Expected Output Statistics

```
✓ Correlation:              > 0.98 (typically 0.999)
✓ Mean absolute error:      < 1% (typically 0.3%)
✓ Accuracy within ±1%:      > 80% (typically 90%+)
✓ Accuracy within ±5%:      > 95% (typically 99%+)
✓ Error distribution:       Centered at zero
✓ Robustness:               Consistent across price ranges
```

If not meeting these targets, check:
1. Data extraction correctness
2. V_bound formula implementation
3. Simulation result format
4. Outlier filtering

---

**Complete Reference Available**: `RESEARCH_VISUALIZATIONS_GUIDE.md`

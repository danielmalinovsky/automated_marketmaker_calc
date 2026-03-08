"""Volume bound visualization for AMM price movements."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional


def plot_volume_bound(single_path: pd.DataFrame, 
                      figsize: Tuple[int, int] = (12, 7),
                      ax: Optional[plt.Axes] = None,
                      denominator: str = 'y') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot theoretical volume bound vs simulation steps.
    
    This visualization shows the relationship between price movements and the 
    minimum volume required to execute those moves in a constant product AMM.
    
    Args:
        single_path: DataFrame from payoff.paths_df['dfs'] containing simulation data
        figsize: Figure size as (width, height)
        ax: Matplotlib axes object. If None, creates new figure
        denominator: 'x' or 'y' - which asset to denominate volume in ('y' by default)
        
    Returns:
        Tuple of (fig, ax) - the matplotlib figure and axes objects
        
    Raises:
        KeyError: If required columns are missing from single_path
        ValueError: If data is invalid or insufficient
        
    Example:
        >>> single_path = payoff.paths_df['dfs'].at[0]
        >>> fig, ax = plot_volume_bound(single_path, denominator='y')
        >>> plt.show()
    """
    # Validate denominator parameter
    if denominator not in ['x', 'y']:
        raise ValueError(f"denominator must be 'x' or 'y', got '{denominator}'")
    
    # Set column names based on denominator
    if denominator == 'y':
        amount_col = 'amount_y_pool'
        volume_col = 'V_required_y_vol'
        denom_label = 'Y'
    else:  # denominator == 'x'
        amount_col = 'amount_x_pool'
        volume_col = 'V_required_vol'
        denom_label = 'X'
    
    # Validate input data
    required_cols = ['FX', amount_col, volume_col]
    missing_cols = [col for col in required_cols if col not in single_path.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Calculate FX price ratios (price change at each step)
    single_path_copy = single_path.copy()
    single_path_copy['FX_ratio_diff'] = single_path_copy['FX'] / single_path_copy['FX'].shift(1)
    
    # Align data by dropping NaNs from both columns simultaneously
    valid_mask = single_path_copy['FX_ratio_diff'].notna() & single_path_copy[volume_col].notna()
    price_ratios = single_path_copy.loc[valid_mask, 'FX_ratio_diff'].values
    v_required = single_path_copy.loc[valid_mask, volume_col].values
    
    if len(price_ratios) == 0:
        raise ValueError("No valid FX ratio differences found in data")
    
    # Create theoretical volume bound curves
    # Initial state
    price_ratios_range = np.linspace(price_ratios.min(), price_ratios.max(), num=100)
    reserve_initial = single_path_copy[amount_col].iloc[0]  # Initial reserves
    reserve_final = single_path_copy[amount_col].iloc[-1]  # Final reserves
    
    if reserve_initial <= 0:
        raise ValueError(f"Initial {denom_label} reserves must be positive, got {reserve_initial}")
    if reserve_final <= 0:
        raise ValueError(f"Final {denom_label} reserves must be positive, got {reserve_final}")
    
    V_bound_initial = reserve_initial * np.abs(1 - 1.0 / np.sqrt(price_ratios_range))
    V_bound_final = reserve_final * np.abs(1 - 1.0 / np.sqrt(price_ratios_range))
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot theoretical bounds
    ax.plot(price_ratios_range, V_bound_initial,
            'b-', linewidth=3, label='Theoretical Volume Bound (Initial)', zorder=3)
    ax.plot(price_ratios_range, V_bound_final,
            'r-', linewidth=3, label='Theoretical Volume Bound (Final)', zorder=3)
    
    # Plot simulation steps as scatter with colorbar
    scatter = ax.scatter(price_ratios, v_required,
                        s=80, alpha=0.6, c=np.arange(len(price_ratios)),
                        cmap='viridis', edgecolors='black', linewidth=0.5,
                        label='Simulation Steps', zorder=4)
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, ax=ax, label='Simulation Step Index')
    cbar.set_label('Simulation Step Index\n(earlier → later)', fontsize=11, fontweight='bold')
    
    # Add reference line at P/P₀ = 1 (initial price)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Initial Price (P/P₀ = 1)', zorder=2)
    
    # Add labels and formatting with detailed statistics in title
    price_std = single_path_copy['FX'].std()
    volume_std = single_path_copy[volume_col].std()
    k_pool_initial = single_path_copy['k_pool'].iloc[0] if 'k_pool' in single_path_copy.columns else 0
    k_pool_final = single_path_copy['k_pool'].iloc[-1] if 'k_pool' in single_path_copy.columns else 0
    
    title_text = (
        f'Path-Independent Volume for Price Movement in Constant Product AMM (Denominated in {denom_label})\n'
        f'Price σ = {price_std:.4f} | Volume σ = {volume_std:.2f} | '
        f'k_pool: {k_pool_initial:.0f} → {k_pool_final:.0f}'
    )
    
    ax.set_xlabel('Price Ratio (P/P₀)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Minimum Volume Required ({denom_label} denominated)', fontsize=12, fontweight='bold')
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
    
    # Improve legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')
    
    # Enhance grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Improve layout
    fig.tight_layout()
    
    return fig, ax

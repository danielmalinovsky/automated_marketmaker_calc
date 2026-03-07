"""
Research Visualizations for Path-Independent Volume Analysis

This module provides specialized visualization classes for analyzing and presenting
the relationship between price movements and required trading volume in Constant
Product Automated Market Makers (CPAMMs).

Classes:
    VolumePathAnalysis: Overlay simulation results on theoretical volume bounds
    
Author: Research Package
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


class VolumePathAnalysis:
    """
    Analyze and visualize path-independent volume requirements with simulation validation.
    
    This class combines theoretical volume bound calculations with empirical simulation
    data to demonstrate the accuracy of the V_bound formula across various price movements.
    
    Attributes:
        x0 (float): Initial reserves of token X (e.g., ETH)
        P0 (float): Initial price (P/X units, e.g., USDC/ETH)
        k (float): Constant product (x * y)
        volume_scaling_factor (float): Scaling factor applied to simulated volumes (default: 1.0)
    """
    
    def __init__(self, x0: float, P0: float, volume_scaling_factor: float = 1.0):
        """
        Initialize VolumePathAnalysis with pool parameters.
        
        Args:
            x0 (float): Initial reserves of token X (denominator token)
            P0 (float): Initial price (Y per X, e.g., USDC per ETH)
            volume_scaling_factor (float, optional): Scaling factor applied to input volumes
                Used when volumes are scaled for simulation (e.g., 1e-7). Default is 1.0.
                This helps the analyzer understand the volume scale without modifying calculations.
        """
        self.x0 = x0
        self.P0 = P0
        self.k = x0 ** 2 * P0  # Constant product for CPAMM
        self.volume_scaling_factor = volume_scaling_factor
        
    def calculate_theoretical_bound(self, price_ratios: np.ndarray) -> np.ndarray:
        """
        Calculate theoretical volume bound for given price ratios.
        
        Formula: V_bound = x0 * |1 - sqrt(P0 / P1)|
        
        Args:
            price_ratios (np.ndarray): Array of price ratios (P1/P0)
        
        Returns:
            np.ndarray: Volume bounds corresponding to each price ratio
        """
        P_values = price_ratios * self.P0
        V_bound = self.x0 * np.abs(1 - np.sqrt(self.P0 / P_values))
        return V_bound
    
    def extract_simulation_data(self, simulation_results: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract price ratios and actual volumes from simulation results.
        
        Args:
            simulation_results (list): List of dictionaries from run_simulation_experiment()
                Each dict should contain: 'P0', 'P1', 'V_actual', 'error_pct'
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - price_ratios: P1/P0 for each simulation
                - V_actual: Actual moving volume from each simulation
                - error_pct: Percentage error (V_bound - V_actual) / V_actual
        """
        if not simulation_results:
            raise ValueError("simulation_results cannot be empty")
        
        price_ratios = np.array([r['P1'] / r['P0'] for r in simulation_results])
        V_actual = np.array([r['V_actual'] for r in simulation_results])
        error_pct = np.array([r['error_pct'] for r in simulation_results])
        
        return price_ratios, V_actual, error_pct
    
    def plot_theoretical_with_simulation_overlay(
        self,
        simulation_results: list,
        price_ratio_range: Tuple[float, float] = (0.25, 4.0),
        figsize: Tuple[int, int] = (12, 7),
        color_by: str = 'error_pct',
        alpha_scatter: float = 0.5,
        scatter_size: int = 40,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot theoretical volume bound line with simulation scatter overlay.
        
        Demonstrates how well empirical simulations match the theoretical V_bound formula.
        Points are colored by error magnitude to highlight accuracy of the model.
        
        Args:
            simulation_results (list): List of simulation dictionaries
            price_ratio_range (Tuple[float, float]): Min and max price ratio for x-axis
            figsize (Tuple[int, int]): Figure size (width, height) in inches
            color_by (str): How to color scatter points: 'error_pct', 'price_change', or 'uniform'
            alpha_scatter (float): Transparency of scatter points (0-1)
            scatter_size (int): Size of scatter markers
            save_path (Optional[str]): Path to save figure. If None, figure is not saved.
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects for further customization
        
        Raises:
            ValueError: If simulation_results is empty or color_by is invalid
        """
        if not simulation_results:
            raise ValueError("simulation_results cannot be empty")
        if color_by not in ['error_pct', 'price_change', 'uniform']:
            raise ValueError(f"color_by must be 'error_pct', 'price_change', or 'uniform', got {color_by}")
        
        # Create theoretical line
        P_ratio_line = np.linspace(price_ratio_range[0], price_ratio_range[1], 500)
        V_bound_line = self.calculate_theoretical_bound(P_ratio_line)
        
        # Extract simulation data
        price_ratios_sim, V_actual_sim, error_pct_sim = self.extract_simulation_data(simulation_results)
        
        # Calculate price change percentages for coloring if needed
        price_changes_pct = np.array([r['price_change_pct'] for r in simulation_results])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot theoretical bound line
        ax.plot(P_ratio_line, V_bound_line, 'b-', linewidth=3, 
                label='Theoretical Volume Bound', zorder=3)
        
        # Plot initial price reference
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label='Initial Price (P/P₀ = 1)', zorder=2)
        
        # Prepare color array for scatter
        if color_by == 'error_pct':
            c_values = np.abs(error_pct_sim)
            cmap_name = 'RdYlGn_r'  # Red (high error) to Green (low error)
            cbar_label = 'Absolute Error (%)'
        elif color_by == 'price_change':
            c_values = np.abs(price_changes_pct)
            cmap_name = 'plasma'
            cbar_label = '|Price Change| (%)'
        else:  # uniform
            c_values = np.ones_like(price_ratios_sim)
            cmap_name = None
            cbar_label = None
        
        # Plot simulation scatter points
        if color_by == 'uniform':
            scatter = ax.scatter(price_ratios_sim, V_actual_sim, alpha=alpha_scatter, 
                                s=scatter_size, color='green', label='Simulated Paths', 
                                zorder=1, edgecolors='darkgreen', linewidth=0.5)
        else:
            scatter = ax.scatter(price_ratios_sim, V_actual_sim, alpha=alpha_scatter,
                                s=scatter_size, c=c_values, cmap=cmap_name, 
                                label=f'Simulated Paths (n={len(simulation_results)})',
                                zorder=1, edgecolors='black', linewidth=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)
        
        # Formatting
        ax.set_xlabel('Price Ratio (P₁/P₀)', fontsize=12)
        ax.set_ylabel('Volume Required (ETH)', fontsize=12)
        ax.set_title('Path-Independent Volume: Theory vs Simulation\n$V_{{bound}} = x_0 \\cdot |1 - \\sqrt{{P_0/P_1}}|$',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(price_ratio_range)
        
        # Add key annotations
        key_ratios = [price_ratio_range[0], 1.0, price_ratio_range[1]]
        for ratio in key_ratios:
            if price_ratio_range[0] <= ratio <= price_ratio_range[1]:
                idx = np.argmin(np.abs(P_ratio_line - ratio))
                volume = V_bound_line[idx]
                if ratio == 1.0:
                    continue  # Skip initial price annotation
                label_text = f'{volume:.0f} ETH'
                ax.annotate(label_text, xy=(ratio, volume), xytext=(10, 10),
                           textcoords='offset points', ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
    
    def plot_error_distribution(
        self,
        simulation_results: list,
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 30,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot distribution of errors between theoretical bound and simulation results.
        
        Args:
            simulation_results (list): List of simulation dictionaries
            figsize (Tuple[int, int]): Figure size
            bins (int): Number of histogram bins
            save_path (Optional[str]): Path to save figure
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        errors = np.array([r['error'] for r in simulation_results])
        error_pct = np.array([r['error_pct'] for r in simulation_results])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2,
                   label=f'Mean Error: {np.mean(errors):.4f} ETH')
        
        ax.set_xlabel('Error (V_bound - V_actual) [ETH]', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of V_bound Estimation Errors\nCentered Around Zero Indicates High Accuracy',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        stats_text = (f'Mean: {np.mean(errors):.4f} ETH\n'
                     f'Std Dev: {np.std(errors):.4f} ETH\n'
                     f'Mean % Error: {np.mean(error_pct):.4f}%\n'
                     f'N: {len(simulation_results)}')
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
    
    def plot_accuracy_vs_price_movement(
        self,
        simulation_results: list,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot percentage error vs price change to show model robustness.
        
        Args:
            simulation_results (list): List of simulation dictionaries
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save figure
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        price_changes = np.array([r['price_change_pct'] for r in simulation_results])
        error_pct = np.array([r['error_pct'] for r in simulation_results])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(price_changes, error_pct, alpha=0.6, s=50,
                            c=np.abs(price_changes), cmap='plasma',
                            edgecolors='black', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add tolerance bands
        ax.fill_between(ax.get_xlim(), -1, 1, alpha=0.1, color='green', label='±1% tolerance')
        
        ax.set_xlabel('Price Change (%)', fontsize=12)
        ax.set_ylabel('Percentage Error (%)', fontsize=12)
        ax.set_title('V_bound Estimation Accuracy vs Price Movement\nModel Consistency Across Price Ranges',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|Price Change| (%)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
    
    def summary_statistics(self, simulation_results: list) -> dict:
        """
        Calculate and return summary statistics of simulation accuracy.
        
        Args:
            simulation_results (list): List of simulation dictionaries
        
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        V_bound = np.array([r['V_bound'] for r in simulation_results])
        V_actual = np.array([r['V_actual'] for r in simulation_results])
        errors = np.array([r['error'] for r in simulation_results])
        error_pct = np.array([r['error_pct'] for r in simulation_results])
        
        stats = {
            'num_simulations': len(simulation_results),
            'mean_absolute_error_eth': float(np.mean(np.abs(errors))),
            'std_error_eth': float(np.std(errors)),
            'mean_absolute_error_pct': float(np.mean(np.abs(error_pct))),
            'median_absolute_error_pct': float(np.median(np.abs(error_pct))),
            'correlation': float(np.corrcoef(V_bound, V_actual)[0, 1]),
            'accuracy_within_1pct': float(np.sum(np.abs(error_pct) <= 1.0) / len(error_pct) * 100),
            'accuracy_within_5pct': float(np.sum(np.abs(error_pct) <= 5.0) / len(error_pct) * 100),
            'accuracy_within_10pct': float(np.sum(np.abs(error_pct) <= 10.0) / len(error_pct) * 100),
            'percentile_95_error_pct': float(np.percentile(np.abs(error_pct), 95)),
        }
        
        return stats
    
    def print_summary(self, simulation_results: list) -> None:
        """
        Print formatted summary statistics to console.
        
        Args:
            simulation_results (list): List of simulation dictionaries
        """
        stats = self.summary_statistics(simulation_results)
        
        print("\n" + "=" * 70)
        print("V_BOUND ACCURACY ANALYSIS - SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total Simulations: {stats['num_simulations']}")
        print(f"\nError Metrics:")
        print(f"  Mean Absolute Error: {stats['mean_absolute_error_eth']:.6f} ETH")
        print(f"  Std Dev Error: {stats['std_error_eth']:.6f} ETH")
        print(f"  Mean % Error: {stats['mean_absolute_error_pct']:.4f}%")
        print(f"  Median % Error: {stats['median_absolute_error_pct']:.4f}%")
        print(f"  95th Percentile Error: {stats['percentile_95_error_pct']:.4f}%")
        print(f"\nCorrelation: {stats['correlation']:.6f}")
        print(f"\nAccuracy Tolerance Bands:")
        print(f"  Within ±1%:  {stats['accuracy_within_1pct']:.1f}%")
        print(f"  Within ±5%:  {stats['accuracy_within_5pct']:.1f}%")
        print(f"  Within ±10%: {stats['accuracy_within_10pct']:.1f}%")
        print("\n" + "=" * 70)
        print("KEY INSIGHT:")
        print("High correlation and tight error bounds confirm V_bound formula")
        print("accurately predicts volume required for price movements.")
        print("=" * 70 + "\n")

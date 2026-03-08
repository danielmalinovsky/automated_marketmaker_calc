# from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics
from IPython.display import clear_output
import time
import datetime
from typing import Optional, Union, Tuple, List, Dict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StochasticPlotter:
    """Utility class for plotting stochastic processes."""
    
    @staticmethod
    def plot_process_paths(sim_df: pd.DataFrame, predicted_period: int, 
                          backtesting: bool = False, 
                          process_type: str = 'GBM',
                          mu: float = None, sigma: float = None,
                          theta: float = None, S0: float = None) -> None:
        """
        Unified plotting function for stochastic processes.
        
        Args:
            sim_df: DataFrame with simulated paths
            predicted_period: Number of periods predicted
            backtesting: Whether this is a backtest
            process_type: 'GBM' or 'OUP'
            mu, sigma, theta: Process parameters
            S0: Starting value
        """
        if S0 is None:
            S0 = sim_df.iloc[0, 0] if not sim_df.empty else 0
        
        # Create time arrays
        if backtesting:
            time_space = np.linspace(0, 1, predicted_period + 1)
            plot_df = sim_df
        else:
            time_space = np.linspace(0, 1, 2 * predicted_period + 1)
            df_nan = pd.DataFrame(np.nan, index=range(predicted_period), 
                                 columns=range(len(sim_df.columns)))
            plot_df = pd.concat([df_nan, sim_df]).reset_index(drop=True)
        
        # Calculate statistics
        mean_path = plot_df.mean(axis=1)
        median_path = plot_df.median(axis=1)
        confidence_95 = np.percentile(plot_df.values, 95, axis=1)
        confidence_05 = np.percentile(plot_df.values, 5, axis=1)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        # --- Top plot: Paths ---
        # Plot confidence intervals
        axes[0].fill_between(time_space[:len(plot_df)], confidence_05, confidence_95, 
                            alpha=0.2, color='skyblue', label='90% Confidence Interval')
        
        # Plot individual paths
        axes[0].plot(time_space[:len(plot_df)], plot_df.values, alpha=0.1, 
                    color='blue', linewidth=0.3)
        
        # Plot mean and median
        axes[0].plot(time_space[:len(plot_df)], mean_path, 'k-', linewidth=3, 
                    label='Mean path')
        axes[0].plot(time_space[:len(plot_df)], median_path, 'r--', linewidth=2, 
                    label='Median path')
        
        # Process-specific elements
        if process_type == 'GBM':
            axes[0].axhline(y=S0, color='g', linestyle=':', linewidth=2, 
                           label=f'S₀ = {S0:.2f}')
            # Theoretical mean for GBM
            if mu is not None:
                theoretical_mean = S0 * np.exp(mu * time_space[:len(plot_df)])
                axes[0].plot(time_space[:len(plot_df)], theoretical_mean, 'm:', 
                            linewidth=2, label=f'S₀exp(μt)')
            
        elif process_type == 'OUP':
            if mu is not None:
                axes[0].axhline(y=mu, color='orange', linestyle='--', linewidth=2, 
                               label=f'Long-term mean (μ) = {mu:.2f}')
            axes[0].axhline(y=S0, color='g', linestyle=':', linewidth=2, 
                           label=f'V₀ = {S0:.2f}')
        
        # Add vertical separator for future predictions
        if not backtesting:
            axes[0].axvline(x=0.5, color='gray', linestyle=':', linewidth=2)
            axes[0].text(0.51, axes[0].get_ylim()[0] + 0.05 * 
                        (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), 
                        'Future', rotation=90, verticalalignment='bottom')
        
        # Titles and labels
        if process_type == 'GBM':
            eqn = f"$dS_t = \mu S_t dt + \sigma S_t dW_t$"
            params = f"$S_0 = {S0:.4f}, \mu = {mu:.4f}, \sigma = {sigma:.4f}$"
        else:  # OUP
            eqn = f"$dV_t = \\theta(\mu - V_t)dt + \sigma dW_t$"
            params = f"$V_0 = {S0:.2f}, \mu = {mu:.2f}, \sigma = {sigma:.2f}, \\theta = {theta:.2f}$"
        
        axes[0].set_title(f"{process_type} Process Simulation\n{eqn}\n{params}", 
                         fontsize=14, pad=20)
        axes[0].set_ylabel(f"{'Price' if process_type == 'GBM' else 'Volume'} $(S_t)$", 
                          fontsize=12)
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # --- Bottom plot: Distribution at final time ---
        final_values = sim_df.iloc[-1, :].values
        axes[1].hist(final_values, bins=50, alpha=0.7, color='steelblue', 
                    edgecolor='black', density=True)
        
        # Add vertical lines for key statistics
        axes[1].axvline(x=np.mean(final_values), color='red', linestyle='-', 
                       linewidth=2, label=f'Mean: {np.mean(final_values):.2f}')
        axes[1].axvline(x=np.median(final_values), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(final_values):.2f}')
        axes[1].axvline(x=S0, color='orange', linestyle=':', linewidth=2, 
                       label=f'Initial: {S0:.2f}')
        
        axes[1].set_xlabel(f"Final {('Price' if process_type == 'GBM' else 'Volume')} Distribution", 
                          fontsize=12)
        axes[1].set_ylabel("Density", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()


class PriceSim:
    """
    Optimized version of price_sim class for simulating financial price data using Geometric Brownian Motion (GBM).
    Changes made:
    1. Renamed to follow Python naming conventions (PriceSim instead of price_sim)
    2. Optimized data loading and calculations
    3. Reduced redundant code
    4. Added type hints for better documentation
    5. Improved error handling
    """

    def __init__(self):
        """Initialize PriceSim with default values."""
        self.ticker = ''
        self.start_date = []
        self.end_date = []
        self.interval = ''
        self.predicted_period = 0
        self.backtesting = False
        self.FX_data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.mu = 0.0
        self.sigma = 0.0
        self.sim = pd.DataFrame()
        self.S0 = 0.0

    def data_loading(self, ticker: str, start_date: List[int], 
                    end_date: Union[List[int], str], interval: str) -> pd.DataFrame:
        """
        Load historical price data from Yahoo Finance.
        
        Optimizations:
        1. Combined date parsing logic
        2. Removed redundant self assignments
        3. Added error handling for network requests
        """
        try:
            # Parse start date
            yh_start_date = int(time.mktime(
                datetime.datetime(start_date[0], start_date[1], start_date[2], 23, 59).timetuple()
            ))
            
            # Parse end date
            if end_date == 'today':
                yh_end_date = int(time.mktime(datetime.datetime.now().timetuple()))
            else:
                yh_end_date = int(time.mktime(
                    datetime.datetime(end_date[0], end_date[1], end_date[2], 23, 59).timetuple()
                ))
            
            # Construct query URL
            query_string = (
                f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}'
                f'?period1={yh_start_date}&period2={yh_end_date}'
                f'&interval={interval}&events=history&includeAdjustedClose=true'
            )
            
            # Load data
            FX_data = pd.read_csv(query_string)
            FX_data['Adj Close'] = FX_data['Adj Close']  # This line seems redundant but keeping for compatibility
            
            return FX_data
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def returnify(self, FX_data: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate simple returns.
        
        Optimization: Simplified logic using vectorized operations
        """
        if date_col is not None:
            prices = FX_data.loc[:, FX_data.columns != date_col]
            dates = FX_data[date_col]
            returns = prices.pct_change()  # More efficient than manual calculation
            returnified = pd.concat([dates, returns], axis=1)
        else:
            returnified = FX_data.pct_change()  # Vectorized operation
        
        return returnified

    def log_returnify(self, FX_data: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate log returns.
        
        Optimization: Simplified logic using vectorized operations
        """
        if date_col is not None:
            prices = FX_data.loc[:, FX_data.columns != date_col]
            dates = FX_data[date_col]
            returns = np.log(prices).diff()  # Vectorized log difference
            returnified = pd.concat([dates, returns], axis=1)
        else:
            returnified = np.log(FX_data).diff()  # Vectorized operation
        
        return returnified

    def GBM_params(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate mean and standard deviation of returns."""
        # Using numpy for faster calculations
        mu = np.mean(returns.values)
        sigma = np.std(returns.values, ddof=1)  # Sample standard deviation
        
        return mu, sigma

    def GBM(self, mu: float, sigma: float, S0: float, steps: int, 
            n_paths: int, plot: str = 'N', pandas: str = 'Y') -> Union[pd.DataFrame, np.ndarray]:
        """
        Simulate Geometric Brownian Motion.
        
        Optimizations:
        1. Vectorized all calculations
        2. Pre-allocated arrays
        3. Removed redundant intermediate variables
        """
        T = 1.0
        dt = T / steps
        
        # Pre-allocate array for efficiency
        paths_shape = (n_paths, steps)
        
        # Vectorized simulation using numpy
        # Generate all random numbers at once
        random_numbers = np.random.normal(0, np.sqrt(dt), size=paths_shape).T
        
        # Calculate increments using vectorized operations
        increments = np.exp((mu - sigma ** 2 / 2) * dt + sigma * random_numbers)
        
        # Initialize with ones and multiply
        St = np.vstack([np.ones(n_paths), increments])
        St = S0 * St.cumprod(axis=0)
        
        # Plot if requested
        if plot == 'Y':
            time_points = np.linspace(0, T, steps + 1)
            tt = np.full((n_paths, steps + 1), time_points).T
            
            plt.figure(figsize=(10, 6))
            plt.plot(tt, St)
            plt.xlabel("Years $(t)$")
            plt.ylabel("Stock Price $(S_t)$")
            plt.title(f"Realizations of Geometric Brownian Motion\n"
                     f"$dS_t = \mu S_t dt + \sigma S_t dW_t$\n"
                     f"$S_0 = {S0:.4f}, \mu = {mu:.6f}, \sigma = {sigma:.6f}$")
            plt.grid(True)
            plt.show()
        
        # Return appropriate format
        return pd.DataFrame(St) if pandas == 'Y' else St

    def unified_plot(self, sim_df: pd.DataFrame, predicted_period: int, 
               backtesting: bool = False) -> None:
        """Plot GBM paths using enhanced plotting."""
        StochasticPlotter.plot_process_paths(
            sim_df=sim_df,
            predicted_period=predicted_period,
            backtesting=backtesting,
            process_type='GBM',
            mu=self.mu,
            sigma=self.sigma,
            S0=sim_df.iloc[0, 0] if not sim_df.empty else 0
        )

    def pipeline(self, predicted_period: int, FX_data: pd.DataFrame = pd.DataFrame(), n_paths: int = 10000,
                ticker: str = '', start_date: List[int] = [], 
                end_date: Union[List[int], str] = [], interval: str = '',
                backtesting: bool = True, plot_sim: bool = False) -> None:
        """
        Execute complete GBM simulation pipeline.
        
        Optimizations:
        1. Reduced redundant calculations
        2. Better error handling
        3. More efficient data flow
        """
        # Store parameters
        self.predicted_period = predicted_period
        self.backtesting = backtesting
        
        # Load data if not provided
        if FX_data.empty:
            if not ticker or not start_date:
                raise ValueError("Either provide FX_data or ticker with start_date")
            self.FX_data = self.data_loading(ticker, start_date, end_date, interval)
        else:
            self.FX_data = FX_data
        
        # Check if we have enough data
        if len(self.FX_data) < predicted_period + 1:
            raise ValueError(f"Not enough data. Need at least {predicted_period + 1} data points.")
        
        # Calculate returns
        self.returns = self.log_returnify(FX_data=self.FX_data, date_col='Date')
        
        # Remove NaN values from returns
        valid_returns = self.returns['Adj Close'].dropna()
        if len(valid_returns) == 0:
            raise ValueError("No valid returns calculated")
        
        # Calculate GBM parameters
        self.mu, self.sigma = self.GBM_params(valid_returns)
        
        # Determine starting price
        if backtesting:
            start_idx = len(self.FX_data) - predicted_period - 1
        else:
            start_idx = len(self.FX_data) - 1
        
        # Check index validity
        if start_idx < 0 or start_idx >= len(self.FX_data):
            raise IndexError(f"Invalid start index: {start_idx}")
        
        S0 = self.FX_data.at[start_idx, 'Adj Close']
        
        # Scale parameters for prediction period
        mu_scaled = self.mu * predicted_period
        sigma_scaled = self.sigma * np.sqrt(predicted_period)
        
        # Run simulation
        self.sim = self.GBM(
            mu=mu_scaled,
            sigma=sigma_scaled,
            S0=S0,
            steps=predicted_period,
            n_paths=n_paths,
            plot='N',
            pandas='Y'
        )
        
        # Plot if requested
        if plot_sim:

            self.unified_plot(
                sim_df=self.sim,
                predicted_period=predicted_period,
                backtesting=backtesting
            )

        return self.sim


class VolumeSim:
    """
    Optimized version of volume_sim class.
    
    Changes made:
    1. Reused methods from PriceSim to avoid duplication
    2. Vectorized UOP simulation
    3. Better memory management
    """
    
    # Reuse methods from PriceSim to avoid code duplication
    returnify = PriceSim.returnify
    log_returnify = PriceSim.log_returnify
    GBM_params = PriceSim.GBM_params
    GBM = PriceSim.GBM
    unified_plot = PriceSim.unified_plot


    def UOP(self, mu: float, sigma: float, theta: float, S0: float, 
            steps: int, n_paths: int, plot: str = 'N', 
            pandas: str = 'Y') -> Union[pd.DataFrame, np.ndarray]:
        """
        Simulate Ornstein-Uhlenbeck Process.
        
        Optimizations:
        1. Vectorized the simulation
        2. Removed loop for path generation
        3. Pre-allocated array
        """
        T = 1.0
        dt = T / steps
        
        # Pre-allocate array for all paths
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = S0
        
        # Vectorized simulation
        for i in range(steps):
            # Generate all random increments at once
            dw = np.random.normal(scale=np.sqrt(dt), size=n_paths)
            # Vectorized update
            paths[:, i + 1] = paths[:, i] + theta * (mu - paths[:, i]) * dt + sigma * dw
        
        # Plot if requested
        if plot == 'Y':
            time_steps = np.linspace(0, T, steps + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, paths.T, alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Ornstein-Uhlenbeck Process with Multiple Paths')
            plt.grid(True)
            plt.show()
        
        # Return appropriate format
        return pd.DataFrame(paths.T) if pandas == 'Y' else paths.T

    def unified_plot(self, sim_df: pd.DataFrame, predicted_period: int, 
                   backtesting: bool = False) -> None:
        """Plot UOP paths using enhanced plotting."""
        StochasticPlotter.plot_process_paths(
            sim_df=sim_df,
            predicted_period=predicted_period,
            backtesting=backtesting,
            process_type='OUP',
            mu=self.mu,
            sigma=self.sigma,
            theta=self.theta,
            S0=sim_df.iloc[0, 0] if not sim_df.empty else 0
        )

    def pipeline(self, volume: pd.Series, theta: float, predicted_period: int, n_paths: int = 10000,
                backtesting: bool = True, plot_sim: bool = False) -> pd.DataFrame:
        """
        Execute UOP simulation pipeline.
        
        Optimizations:
        1. Simplified parameter calculation
        2. Better error handling
        """
        # Check input data
        if volume.empty:
            raise ValueError("Volume data cannot be empty")
        
        if len(volume) < predicted_period + 1:
            raise ValueError(f"Need at least {predicted_period + 1} volume data points")
        
        # Calculate parameters from volume data (not returns as in original)
        # Original code used volume directly, not returns
        self.theta = theta
        self.mu, self.sigma = self.GBM_params(volume)
        # self.mu = np.mean(volume.values)
        # self.sigma = np.std(volume.values, ddof=1)
        
        # Determine starting value
        if backtesting:
            # For backtesting: we simulate the historical period
            # Start from predicted_period+1 steps before the end
            start_idx = len(volume) - predicted_period - 1
            if start_idx < 0:
                raise ValueError(f"Not enough data for backtesting. Need at least {predicted_period + 1} points.")
        else:
            # For forecasting: start from the last available data point
            start_idx = len(volume) - 1
        
        if start_idx < 0 or start_idx >= len(volume):
            raise IndexError(f"Invalid start index: {start_idx}")
        
        S0 = volume.iloc[start_idx]
        
        # Run UOP simulation
        self.sim = self.UOP(
            mu=self.mu,
            sigma=self.sigma,
            theta=self.theta,
            S0=S0,
            steps=predicted_period,
            n_paths=n_paths,
            plot='N',
            pandas='Y'
        )
        
        # Add plotting option
        if plot_sim:
            self.unified_plot(
                sim_df=self.sim,
                predicted_period=predicted_period,
                backtesting=backtesting
            )
        
        return self.sim


class Payoff:
    """
    Optimized Payoff class for AMM calculations with volume-arbitrage decomposition.
    
    Changes made:
    1. Added volume-arbitrage decomposition logic
    2. Added path filtering based on volume sufficiency
    3. Added detailed fee allocation based on arbitrage direction
    4. Added logging for dropped paths
    """
    
    def __init__(self):
        """Initialize data structures."""
        self.pool_reserves = pd.DataFrame(
            columns=['k', 'amount_x', 'amount_y', 'value_in_x', 'x_fee', 'y_fee']
        )
        self.depositor_reserves = pd.DataFrame(
            columns=['k', 'amount_x', 'amount_y', 'value_in_x', 'x_fee', 'y_fee']
        )
        self.depositor_performance = pd.DataFrame(
            columns=['pool_share', 'interest_income', 'hodl_x', 'impermanent_loss', 'IL_rel', 'II_netto']
        )
        self.pool_performance = pd.DataFrame(
            columns=['FX', 'pool_fee', 'volume']
        )
        self.paths_df = pd.DataFrame()
        
        # New attributes for volume-arbitrage decomposition
        self.volume_decomposition = pd.DataFrame(
            columns=['V_total', 'V_required', 'V_required_y', 'V_excess', 'dropped']
        )
        self.dropped_paths_count = 0
        self.dropped_paths_log = []
        
    # ========== Volume-Arbitrage Decomposition Methods ==========
    
    @staticmethod
    def calculate_volume_required(x0: float, P0: float, P1: float) -> float:
        """
        Calculate minimum volume required to move price from P0 to P1 in CPAMM.
        
        Formula: V_required = x0 * |1 - sqrt(P0 / P1)|
        
        Args:
            x0: Initial amount of token X in pool
            P0: Initial price (Y/X)
            P1: Final price (Y/X)
            
        Returns:
            Minimum volume of token X required
        """
        if x0 <= 0:
            return 0.0
        if P0 <= 0 or P1 <= 0:
            return np.inf  # Invalid price
        
        ratio = P0 / P1
        if ratio < 0:  # Negative price ratio
            return np.inf

        return x0 * abs(1 - math.sqrt(ratio))
    
    @staticmethod
    def calculate_volume_excess(V_total: float, V_required: float) -> float:
        """
        Calculate excess volume after arbitrage.
        
        Args:
            V_total: Total daily volume
            V_required: Minimum volume required for price move
            
        Returns:
            Excess volume (V_total - V_required)
        """
        return max(0, V_total - V_required)
    
    @staticmethod
    def calculate_fee_allocation(V_excess: float, V_required: float, 
                                P0: float, P1: float, fee_rate: float) -> Tuple[float, float]:
        """
        Calculate fee allocation between X and Y based on arbitrage direction.
        
        For price increases (P1 > P0): arbitrage sells Y, so V_required fees accrue in Y
        For price decreases (P1 < P0): arbitrage sells X, so V_required fees accrue in X
        Excess volume V_excess assumes balanced flow, fees split 50/50
        
        Note: Fees in Y are normalized by P0 (initial price) to preserve directional
        allocation intent. Using P1 would make Y fees disappear when price moves significantly.
        
        Args:
            V_excess: Excess volume
            V_required: Arbitrage volume
            P0: Initial price (used for normalization)
            P1: Final price (used to determine direction)
            fee_rate: Pool fee rate
            
        Returns:
            Tuple of (fees_in_x, fees_in_y)
        """
        if P1 > P0:  # Price increase: arbitrage sells Y
            fees_in_x = 0.5 * V_excess * fee_rate
            fees_in_y = (0.5 * V_excess + V_required) * fee_rate / P0
        else:  # Price decrease: arbitrage sells X
            fees_in_x = (0.5 * V_excess + V_required) * fee_rate
            fees_in_y = 0.5 * V_excess * fee_rate / P0

        
        return fees_in_x, fees_in_y
    
    def check_volume_sufficiency(self, V_total: float, V_required: float, 
                                tolerance: float = 1e-10) -> bool:
        """
        Check if total volume is sufficient for the price move.
        
        Args:
            V_total: Total daily volume
            V_required: Minimum volume required
            tolerance: Numerical tolerance
            
        Returns:
            True if V_total >= V_required, False otherwise
        """
        return V_total >= V_required - tolerance
    
    def log_dropped_path(self, path_idx: int, time_step: int, 
                        V_total: float, V_required: float, P0: float, P1: float):
        """Log information about dropped paths."""
        self.dropped_paths_count += 1
        self.dropped_paths_log.append({
            'path': path_idx,
            'time_step': time_step,
            'V_total': V_total,
            'V_required': V_required,
            'deficit': V_required - V_total,
            'P0': P0,
            'P1': P1
        })
    
    # ========== Core Calculation Methods ==========
    
    @staticmethod
    def k_product(x: float, y: float) -> float:
        """Calculate product of x and y."""
        return x * y
    
    @staticmethod
    def value_in_x(x: float, y: float, FX: float) -> float:
        """Calculate value in X currency."""
        return x + y * FX
    
    def FX_x_over_y(self, k_pool: float, y_amount: float) -> float:
        """Calculate exchange rate X/Y."""
        if y_amount == 0:
            raise ValueError("y_amount cannot be zero")
        return k_pool / (y_amount ** 2)
    
    def deposit_split(self, deposit_value_in_x: float, percentage_of_x: float, 
                     FX_x_over_y: float) -> Tuple[float, float]:
        """Split deposit into X and Y amounts."""
        x_amount = deposit_value_in_x * percentage_of_x
        y_amount = (deposit_value_in_x - x_amount) / FX_x_over_y
        return x_amount, y_amount
    
    @staticmethod
    def pool_share(k_depositor: float, k_pool: float) -> float:
        """Calculate pool share percentage."""
        if k_pool <= 0:
            return 0.0
        return math.sqrt(k_depositor) / math.sqrt(k_pool)
    
    @staticmethod
    def interest_income(accrued_fee_in_x: float, value_in_x: float) -> float:
        """Calculate interest income as percentage."""
        if value_in_x == 0:
            return 0.0
        return accrued_fee_in_x / value_in_x
    
    @staticmethod
    def hold_in_x(amount_x_t0: float, amount_y_t0: float, FX_tn: float) -> float:
        """Calculate hold value in X currency."""
        return amount_x_t0 + amount_y_t0 * FX_tn
    
    @staticmethod
    def impermanent_loss(FX_in_x_t0: float, FX_in_x_tn: float, relative: str = 'N') -> float:
        """
        Calculate impermanent loss.
        
        For relative='Y': IL = (2*sqrt(r))/(1+r) - 1 where r = FX_tn/FX_t0
        """
        if relative == 'N':
            # This case seems incomplete in original code - value_in_x and hodl_in_x not defined
            # Keeping original structure but this will need to be handled by caller
            raise NotImplementedError("Absolute impermanent loss calculation requires additional parameters")
        else:
            if FX_in_x_t0 == 0:
                return 0.0
            r = FX_in_x_tn / FX_in_x_t0
            return (2 * math.sqrt(r)) / (1 + r) - 1
    
    @staticmethod
    def fee_amount_by_orders(volume_amount_in_origin: Dict[str, float], 
                           fee_rate: float, fee_split: float) -> Tuple[float, float]:
        """Calculate fee amounts from order volumes."""
        x_fee_amount = list(volume_amount_in_origin.values())[0] * fee_rate
        y_fee_amount = list(volume_amount_in_origin.values())[1] * fee_rate
        return x_fee_amount, y_fee_amount
    
    def swap_calc(self, quote_type: str, swap_from_amount_reserve: List[float],
                 swap_to_amount_reserve: List[float]) -> float:
        """Calculate swap amounts."""
        x = swap_from_amount_reserve[2]
        y = swap_to_amount_reserve[2]
        k = self.k_product(x, y)
        
        if quote_type == 'ask':
            y_ask = swap_to_amount_reserve[1]
            if y == y_ask:
                return 0.0
            x_for_ask = (k * (y - y_ask)) / ((y - y_ask) ** 2) - x
            return x_for_ask
        elif quote_type == 'bid':
            x_bid = swap_from_amount_reserve[1]
            if x == x_bid:
                return 0.0
            y_for_bid = y - (k * (x + x_bid)) / ((x + x_bid) ** 2)
            return y_for_bid
        else:
            raise ValueError(f"Invalid quote type: {quote_type}")
    
    def fee_amount_by_reserves(self, FX_in_x: float, V_total: float, 
                              fee_rate: float, fee_split: float = 0.5,
                              x0: float = None,
                              P0: float = None, P1: float = None,
                              use_decomposition: bool = False) -> Tuple[float, float]:
        """
        Calculate fee amounts from reserves with optional volume decomposition.
        
        Args:
            FX_in_x: Current exchange rate (X per Y)
            V_total: Total volume in X terms
            fee_rate: Pool fee rate
            fee_split: Proportion of fees collected in X (0.5 = 50/50 split)
                      Value between 0 and 1. Fees in Y are subject to exchange rate risk.
                      Example: fee_split=0.7 means 70% of fees in X, 30% in Y
            x0: Initial X reserves (required if use_decomposition=True)
            P0: Initial price (required if use_decomposition=True)
            P1: Current price (required if use_decomposition=True)
            use_decomposition: Whether to use volume-arbitrage decomposition
            
        Returns:
            Tuple of (x_fee_amount, y_fee_amount)
        """
        if abs(FX_in_x) < 1e-10: # Prevent division by zero or values too close to zero
            return 0.0, 0.0
        
        if not use_decomposition or x0 is None or P0 is None or P1 is None:
            # Calculate fee amounts with fee_split allocation
            # fee_split determines proportion: X_fee / (X_fee + Y_fee*FX) = fee_split
            y_amount_exchanged = V_total / FX_in_x
            total_traded_value = V_total + y_amount_exchanged
            total_fee_value = total_traded_value * fee_rate
            
            # Allocate fees based on fee_split
            # fee_split * total_fee_value goes to X fees
            # (1-fee_split) * total_fee_value goes to Y fees (as value, converted to Y amount)
            x_fee_amount = total_fee_value * fee_split
            y_fee_amount = total_fee_value * (1 - fee_split) / FX_in_x
            return x_fee_amount, y_fee_amount
        
        # Use volume-arbitrage decomposition
        V_required = self.calculate_volume_required(x0, P0, P1)
        V_excess = self.calculate_volume_excess(V_total, V_required)
        
        # Check if volume is sufficient
        if not self.check_volume_sufficiency(V_total, V_required):
            # Insufficient volume - this path would be dropped
            # Return minimal fees as if only arbitrage happened
            if P1 > P0:
                x_fee_amount = 0
                y_fee_amount = V_required * fee_rate / P1
            else:
                x_fee_amount = V_required * fee_rate
                y_fee_amount = 0
        else:
            # Sufficient volume - use decomposition model
            x_fee_amount, y_fee_amount = self.calculate_fee_allocation(
                V_excess, V_required, P0, P1, fee_rate
            )
        
        return x_fee_amount, y_fee_amount
    
    def reserves_calc(self, k: float, FX_in_x: float, 
                     calculate_x: bool = True) -> float:
        """Calculate reserves amounts."""
        if k <= 0 or FX_in_x <= 0:
            return 0.0
        
        if calculate_x:
            return math.sqrt(k * FX_in_x)
        else:
            return math.sqrt(k / FX_in_x)
    
    @staticmethod
    def profitability_bounds(FX_in_x_t0: float, fee_rate: float) -> Tuple[float, float]:
        """Calculate profitability bounds."""
        y = FX_in_x_t0
        rho = fee_rate
        
        # Check if expression under sqrt is non-negative
        sqrt_term = -rho**2 + 2*rho
        if sqrt_term < 0:
            return np.inf, -np.inf
        
        denominator = -rho**2 - 1 + 2*rho
        if denominator == 0:
            return np.inf, -np.inf
        
        sqrt_val = 2 * y * math.sqrt(sqrt_term)
        numerator_base = -y + y*(rho**2) - 2*y*rho
        
        x1 = (numerator_base + sqrt_val) / denominator
        x2 = (numerator_base - sqrt_val) / denominator
        
        return x1, x2
    
    # ========== Enhanced Time Step Calculations ==========
    
    def t0_calc(self, pool_fee: float, amount_x_pool_t0: float, 
               amount_y_pool_t0: float, total_investment_x: float,
               deposit_split_percentage: float) -> None:
        """Calculate initial time step (t=0).
        
        IMPORTANT: deposit_split_percentage should ALWAYS be 0.5 (50/50 split)
        The pool's token ratio is defined by amount_x_pool_t0 and amount_y_pool_t0.
        The depositor must split their investment proportionally to match this ratio.
        """
        # t0 pool pre-calculation
        self.pool_performance.at[0, 'pool_fee'] = pool_fee
        k_pool_t0 = self.k_product(amount_x_pool_t0, amount_y_pool_t0)
        self.pool_reserves.at[0, 'k'] = k_pool_t0
        
        if amount_y_pool_t0 != 0:
            FX_t0 = self.FX_x_over_y(k_pool_t0, amount_y_pool_t0)
        else:
            FX_t0 = 0.0
        self.pool_performance.at[0, 'FX'] = FX_t0
        
        # t0 depositor pre-calculation
        self.depositor_reserves.at[0, 'value_in_x'] = total_investment_x
        
        # Deposit split - proportional to pool ratio
        deposit_x, deposit_y = self.deposit_split(
            total_investment_x, deposit_split_percentage, FX_t0
        )
        self.depositor_reserves.at[0, 'amount_x'] = deposit_x
        self.depositor_reserves.at[0, 'amount_y'] = deposit_y
        self.depositor_reserves.at[0, 'k'] = self.k_product(deposit_x, deposit_y)
        
        # t0 pool calculation with depositor's funds
        self.pool_reserves.at[0, 'amount_x'] = amount_x_pool_t0 + deposit_x
        self.pool_reserves.at[0, 'amount_y'] = amount_y_pool_t0 + deposit_y
        self.pool_reserves.at[0, 'k'] = self.k_product(
            self.pool_reserves.at[0, 'amount_x'],
            self.pool_reserves.at[0, 'amount_y']
        )
        self.pool_reserves.at[0, 'value_in_x'] = self.value_in_x(
            self.pool_reserves.at[0, 'amount_x'],
            self.pool_reserves.at[0, 'amount_y'],
            FX_t0
        )
        
        # Update FX with new reserves
        if self.pool_reserves.at[0, 'amount_y'] != 0:
            self.pool_performance.at[0, 'FX'] = self.FX_x_over_y(
                self.pool_reserves.at[0, 'k'],
                self.pool_reserves.at[0, 'amount_y']
            )
        
        # Initialize fees to 0
        self.pool_reserves.at[0, 'x_fee'] = 0.0
        self.pool_reserves.at[0, 'y_fee'] = 0.0
        
        # Calculate depositor performance
        self.depositor_performance.at[0, 'pool_share'] = self.pool_share(
            self.depositor_reserves.at[0, 'k'],
            self.pool_reserves.at[0, 'k']
        )
        
        self.depositor_reserves.at[0, 'x_fee'] = 0.0
        self.depositor_reserves.at[0, 'y_fee'] = 0.0
        
        # Calculate interest income
        accrued_fee_value = self.value_in_x(
            self.depositor_reserves.at[0, 'x_fee'],
            self.depositor_reserves.at[0, 'y_fee'],
            self.pool_performance.at[0, 'FX']
        )
        self.depositor_performance.at[0, 'interest_income'] = self.interest_income(
            accrued_fee_value,
            self.depositor_reserves.at[0, 'value_in_x']
        )
        
        # Calculate impermanent loss
        self.depositor_performance.at[0, 'IL_rel'] = self.impermanent_loss(
            self.pool_performance.at[0, 'FX'],
            self.pool_performance.at[0, 'FX'],
            relative='Y'
        )
        
        # Calculate net income
        self.depositor_performance.at[0, 'II_netto'] = (
            self.depositor_performance.at[0, 'interest_income'] + 
            self.depositor_performance.at[0, 'IL_rel']
        )
        
        # Calculate hold value
        self.depositor_performance.at[0, 'hodl_x'] = self.hold_in_x(
            self.depositor_reserves.at[0, 'amount_x'],
            self.depositor_reserves.at[0, 'amount_y'],
            self.pool_performance.at[0, 'FX']
        )
        
        # Initialize volume decomposition tracking
        self.volume_decomposition.at[0, 'V_total'] = 0.0
        self.volume_decomposition.at[0, 'V_required'] = 0.0
        self.volume_decomposition.at[0, 'V_required_y'] = 0.0
        self.volume_decomposition.at[0, 'V_excess'] = 0.0
        self.volume_decomposition.at[0, 'dropped'] = False
    
    def tn_calc(self, FX_timeseries: pd.DataFrame, volume_timeseries: pd.DataFrame, 
            max_paths: int, fee_split: float = 0.5,
            use_volume_decomposition: bool = False,
            drop_insufficient_volume: bool = False) -> None:
        """
        Calculate subsequent time steps with optional volume decomposition.
        
        Args:
            FX_timeseries: DataFrame with FX paths
            volume_timeseries: DataFrame with volume paths
            max_paths: Maximum number of paths to process
            fee_split: Proportion of fees collected in X (0.5 = 50/50 split)
            use_volume_decomposition: Whether to use volume-arbitrage decomposition
            drop_insufficient_volume: Whether to drop paths with insufficient volume
        """
        self.paths_df = pd.DataFrame()
        self.dropped_paths_count = 0
        self.dropped_paths_log = []
        paths_list = []
        
        # Get initial data from t0_calc as templates
        initial_pool_perf = self.pool_performance.copy()
        initial_pool_reserves = self.pool_reserves.copy()
        initial_depositor_reserves = self.depositor_reserves.copy()
        initial_depositor_perf = self.depositor_performance.copy()
        initial_vol_decomp = self.volume_decomposition.copy()
        
        for j in range(max_paths):
            # Skip if column doesn't exist
            if j >= len(FX_timeseries.columns):
                continue
            
            # Reset data structures for this path using copies of initial data
            self.pool_performance = initial_pool_perf.copy().reset_index(drop=True)
            self.pool_reserves = initial_pool_reserves.copy().reset_index(drop=True)
            self.depositor_reserves = initial_depositor_reserves.copy().reset_index(drop=True)
            self.depositor_performance = initial_depositor_perf.copy().reset_index(drop=True)
            self.volume_decomposition = initial_vol_decomp.copy().reset_index(drop=True)
            
            # Track if this path should be dropped
            drop_this_path = False
            drop_steps_in_path = 0
            
            # Get the number of steps for this path (excluding time 0)
            n_steps = min(len(FX_timeseries), len(volume_timeseries))
            
            for i in range(1, n_steps):
                # Skip if data is missing
                if pd.isna(FX_timeseries.iloc[i, j]) or pd.isna(volume_timeseries.iloc[i, j]):
                    # Add placeholder row with NaN values
                    self._add_placeholder_row(i)
                    continue
                
                # CRITICAL FIX: Add new row for this time step BEFORE accessing it
                if i >= len(self.pool_performance):
                    # Initialize new rows with NaN values
                    self.pool_performance.loc[i] = [np.nan, np.nan, np.nan]  # pool_fee, FX, volume
                    self.pool_reserves.loc[i] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    self.depositor_reserves.loc[i] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    self.depositor_performance.loc[i] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    self.volume_decomposition.loc[i] = [np.nan, np.nan, np.nan, np.nan, False]
                
                # Now we can safely set values
                # Copy previous fee rate from time i-1
                fee_rate = self.pool_performance.at[i-1, 'pool_fee']
                self.pool_performance.at[i, 'pool_fee'] = fee_rate
                
                # Get current FX and volume values
                current_FX = FX_timeseries.iloc[i, j]
                V_total = volume_timeseries.iloc[i, j]
                prev_FX = self.pool_performance.at[i-1, 'FX']
                
                # Set current FX and volume
                self.pool_performance.at[i, 'FX'] = current_FX
                self.pool_performance.at[i, 'volume'] = V_total
                
                # IMPROVED: Calculate volume decomposition metrics (always, not just when flag is True)
                # This ensures V_required and V_excess are always available for fee calculation
                if use_volume_decomposition:
                    # Get initial X and Y reserves for volume calculation
                    x0 = self.pool_reserves.at[i-1, 'amount_x']
                    y0 = self.pool_reserves.at[i-1, 'amount_y']
                    V_required = self.calculate_volume_required(x0, prev_FX, current_FX)
                    V_required_y = self.calculate_volume_required(y0, prev_FX, current_FX)
                    
                    # Check volume sufficiency
                    if not self.check_volume_sufficiency(V_total, V_required):
                        if drop_insufficient_volume:
                            # Log and mark this step as dropped
                            self.log_dropped_path(j, i, V_total, V_required, prev_FX, current_FX)
                            drop_steps_in_path += 1
                            
                            # Fill this row with placeholder values and mark as dropped
                            self._fill_dropped_row(i, V_total, V_required, current_FX)
                            continue
                    
                    # Volume is sufficient - calculate excess
                    V_excess = self.calculate_volume_excess(V_total, V_required)
                else:
                    # When decomposition is not used, treat all volume as excess (no minimum required)
                    V_required = 0.0
                    V_required_y = 0.0
                    V_excess = V_total
                
                # Calculate new reserves based on FX change
                prev_k = self.pool_reserves.at[i-1, 'k']
                
                self.pool_reserves.at[i, 'amount_x'] = self.reserves_calc(
                    prev_k, current_FX, calculate_x=True
                )
                self.pool_reserves.at[i, 'amount_y'] = self.reserves_calc(
                    prev_k, current_FX, calculate_x=False
                )
                
                # IMPROVED: Calculate fees with consistent variable availability
                # V_required and V_excess are now guaranteed to be available
                if use_volume_decomposition and V_required > 0:
                    # Use decomposition-based fee allocation
                    x_fee, y_fee = self.calculate_fee_allocation(
                        V_excess, V_required, prev_FX, current_FX, fee_rate
                    )
                    # Store volume decomposition metrics
                    self.volume_decomposition.at[i, 'V_total'] = V_total
                    self.volume_decomposition.at[i, 'V_required'] = V_required
                    self.volume_decomposition.at[i, 'V_required_y'] = V_required_y
                    self.volume_decomposition.at[i, 'V_excess'] = V_excess
                    self.volume_decomposition.at[i, 'dropped'] = False
                else:
                    # Use original fee calculation with fee_split allocation
                    x_fee, y_fee = self.fee_amount_by_reserves(
                        current_FX, V_total, fee_rate,
                        fee_split=fee_split,
                        use_decomposition=False
                    )
                    # Mark no decomposition used (or V_required was 0)
                    if not use_volume_decomposition:
                        self.volume_decomposition.at[i, 'V_total'] = V_total
                        self.volume_decomposition.at[i, 'V_required'] = 0.0
                        self.volume_decomposition.at[i, 'V_required_y'] = 0.0
                        self.volume_decomposition.at[i, 'V_excess'] = V_total
                        self.volume_decomposition.at[i, 'dropped'] = False
                
                self.pool_reserves.at[i, 'x_fee'] = abs(x_fee)
                self.pool_reserves.at[i, 'y_fee'] = abs(y_fee)
                
                # Update reserves with collected fees
                self.pool_reserves.at[i, 'amount_x'] += self.pool_reserves.at[i, 'x_fee']
                self.pool_reserves.at[i, 'amount_y'] += self.pool_reserves.at[i, 'y_fee']
                self.pool_reserves.at[i, 'k'] = self.k_product(
                    self.pool_reserves.at[i, 'amount_x'],
                    self.pool_reserves.at[i, 'amount_y']
                )
                
                # Calculate pool value
                self.pool_reserves.at[i, 'value_in_x'] = self.value_in_x(
                    self.pool_reserves.at[i, 'amount_x'],
                    self.pool_reserves.at[i, 'amount_y'],
                    current_FX
                )
                
                # Depositor calculations
                self.depositor_performance.at[i, 'pool_share'] = self.depositor_performance.at[i-1, 'pool_share']
                
                # Allocate reserves proportionally
                pool_share = self.depositor_performance.at[i, 'pool_share']
                self.depositor_reserves.at[i, 'amount_x'] = self.pool_reserves.at[i, 'amount_x'] * pool_share
                self.depositor_reserves.at[i, 'amount_y'] = self.pool_reserves.at[i, 'amount_y'] * pool_share
                self.depositor_reserves.at[i, 'value_in_x'] = self.pool_reserves.at[i, 'value_in_x'] * pool_share
                
                # Allocate fees
                self.depositor_reserves.at[i, 'x_fee'] = self.pool_reserves.at[i, 'x_fee'] * pool_share
                self.depositor_reserves.at[i, 'y_fee'] = self.pool_reserves.at[i, 'y_fee'] * pool_share
                self.depositor_reserves.at[i, 'k'] = self.k_product(
                    self.depositor_reserves.at[i, 'amount_x'],
                    self.depositor_reserves.at[i, 'amount_y']
                )
                
                # Calculate cumulative interest income
                cum_x_fee = self.depositor_reserves.loc[0:i, 'x_fee'].sum()
                cum_y_fee = self.depositor_reserves.loc[0:i, 'y_fee'].sum()
                cum_fee_value = self.value_in_x(cum_x_fee, cum_y_fee, current_FX)
                
                self.depositor_performance.at[i, 'interest_income'] = self.interest_income(
                    cum_fee_value,
                    self.depositor_reserves.at[0, 'value_in_x']
                )
                
                # Calculate impermanent loss
                initial_FX = self.pool_performance.at[0, 'FX']
                self.depositor_performance.at[i, 'IL_rel'] = self.impermanent_loss(
                    initial_FX,
                    current_FX,
                    relative='Y'
                )
                
                # Calculate net income
                self.depositor_performance.at[i, 'II_netto'] = (
                    self.depositor_performance.at[i, 'interest_income'] + 
                    self.depositor_performance.at[i, 'IL_rel']
                )
                
                # Calculate hold value
                self.depositor_performance.at[i, 'hodl_x'] = self.hold_in_x(
                    self.depositor_reserves.at[i-1, 'amount_x'],
                    self.depositor_reserves.at[i-1, 'amount_y'],
                    current_FX
                )
            
            # Check if we should drop the entire path
            if drop_insufficient_volume and drop_steps_in_path > 0:
                drop_this_path = True
                print(f"Path {j} dropped: {drop_steps_in_path} insufficient volume steps")
            
            # Skip saving this path if it was completely dropped
            if drop_this_path and drop_insufficient_volume:
                continue
            
            # Create merged DataFrame for this path
            merge_df = pd.concat([
                self.pool_performance,
                self.depositor_performance,
                self.pool_reserves.add_suffix('_pool'),
                self.depositor_reserves.add_suffix('_depositor')
            ], axis=1).fillna(0)
            
            # Add volume decomposition if used
            if use_volume_decomposition:
                merge_df = pd.concat([merge_df, self.volume_decomposition.add_suffix('_vol')], axis=1)
            
            # Store in paths DataFrame
            time_step_df = pd.DataFrame({'sim': [j], 'dfs': [merge_df.copy()]})
            paths_list.append(time_step_df)
            
            # Progress update
            if (j + 1) % 10 == 0 or j == max_paths - 1:
                clear_output(wait=True)
                print(f"Progress: {j+1}/{max_paths} paths processed")
                if self.dropped_paths_count > 0:
                    print(f"Dropped steps: {self.dropped_paths_count}")
        
        if paths_list:
            self.paths_df = pd.concat(paths_list, ignore_index=True)
        else:
            self.paths_df = pd.DataFrame(columns=['sim', 'dfs'])
        
        # Final summary
        print(f"\nProcessing complete. Total paths: {len(self.paths_df)}")
        if self.dropped_paths_count > 0:
            print(f"Total dropped steps: {self.dropped_paths_count}")
            if len(self.dropped_paths_log) > 0:
                print("\nDropped steps summary (first 5):")
                for log in self.dropped_paths_log[:5]:
                    print(f"  Path {log['path']}, Step {log['time_step']}: "
                        f"V_total={log['V_total']:.2f}, V_required={log['V_required']:.2f}, "
                        f"deficit={log['deficit']:.2f}")
                if len(self.dropped_paths_log) > 5:
                    print(f"  ... and {len(self.dropped_paths_log) - 5} more")
    
    # ========== Placeholder Method ==========

    def _add_placeholder_row(self, i: int) -> None:
        """Add placeholder row with NaN values for dropped or missing data."""
        # Add rows if they don't exist
        for df, num_cols in [(self.pool_performance, 3), 
                            (self.pool_reserves, 6),
                            (self.depositor_reserves, 6),
                            (self.depositor_performance, 6),
                            (self.volume_decomposition, 5)]:
            if i >= len(df):
                df.loc[i] = [np.nan] * num_cols

    def _fill_dropped_row(self, i: int, V_total: float, V_required: float, current_FX: float) -> None:
        """Fill a row with placeholder values for a dropped step."""
        # Copy values from previous row for continuity
        if i > 0:
            self.pool_performance.loc[i] = self.pool_performance.loc[i-1]
            self.pool_reserves.loc[i] = self.pool_reserves.loc[i-1]
            self.depositor_reserves.loc[i] = self.depositor_reserves.loc[i-1]
            self.depositor_performance.loc[i] = self.depositor_performance.loc[i-1]
        
        # Update FX to current value
        self.pool_performance.at[i, 'FX'] = current_FX
        self.pool_performance.at[i, 'volume'] = V_total
        
        # Set volume decomposition values
        self.volume_decomposition.at[i, 'V_total'] = V_total
        self.volume_decomposition.at[i, 'V_required'] = V_required
        self.volume_decomposition.at[i, 'V_required_y'] = 0.0
        self.volume_decomposition.at[i, 'V_excess'] = 0.0
        self.volume_decomposition.at[i, 'dropped'] = True

    # ========== Analysis Methods ==========
    
    def get_volume_decomposition_summary(self) -> pd.DataFrame:
        """Get summary statistics of volume decomposition."""
        if self.volume_decomposition.empty:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Mean V_total': [self.volume_decomposition['V_total'].mean()],
            'Mean V_required': [self.volume_decomposition['V_required'].mean()],
            'Mean V_excess': [self.volume_decomposition['V_excess'].mean()],
            'V_excess/V_total ratio': [
                self.volume_decomposition['V_excess'].sum() / 
                max(self.volume_decomposition['V_total'].sum(), 1e-10)
            ],
            'Dropped paths': [self.dropped_paths_count]
        })
    
    # ========== Pipeline Method ==========
    
    def pipeline(self, pool_fee: float, amount_x_pool_t0: float, 
                amount_y_pool_t0: float, total_investment_x: float,
                FX_timeseries: pd.DataFrame, volume_timeseries: pd.DataFrame, 
                max_paths: int, fee_split: float = 0.5,
                use_volume_decomposition: bool = False,
                drop_insufficient_volume: bool = False) -> None:
        """Execute complete payoff calculation pipeline.
        
        Args:
            pool_fee: Pool fee rate (e.g., 0.01 for 1%)
            amount_x_pool_t0: Initial X reserves in pool
            amount_y_pool_t0: Initial Y reserves in pool
            total_investment_x: Total investment value in X
            FX_timeseries: Price paths (X/Y rate)
            volume_timeseries: Volume paths
            max_paths: Number of paths to simulate
            fee_split: Proportion of fees collected in X (default 0.5 = 50/50)
                      Value between 0 and 1. Controls how collected fees are split.
                      Example: 0.7 = 70% X fees, 30% Y fees (subject to exchange rate risk)
            use_volume_decomposition: Whether to use directional fee allocation
            drop_insufficient_volume: Whether to drop paths with insufficient volume
        """
        self.fee_split = fee_split
        # IMPORTANT: deposit_split_percentage is now FIXED at 0.5 (50/50 balanced split)
        # The pool's token ratio is already defined by amount_x_pool_t0 and amount_y_pool_t0
        # The depositor must split their investment proportionally to match this ratio
        deposit_split_percentage = 0.5
        self.deposit_split_percentage = deposit_split_percentage
        
        # Initial calculation
        self.t0_calc(
            pool_fee=pool_fee,
            amount_x_pool_t0=amount_x_pool_t0,
            amount_y_pool_t0=amount_y_pool_t0,
            total_investment_x=total_investment_x,
            deposit_split_percentage=deposit_split_percentage
        )
        
        # Time series calculations
        self.tn_calc(
            FX_timeseries=FX_timeseries,
            volume_timeseries=volume_timeseries,
            max_paths=min(max_paths, len(FX_timeseries.columns)),
            fee_split=fee_split,
            use_volume_decomposition=use_volume_decomposition,
            drop_insufficient_volume=drop_insufficient_volume
        )


class Analytics:
    """
    Optimized analytics class with volume decomposition support.
    
    Changes:
    1. Vectorized endpoint calculation
    2. Better memory efficiency
    3. Added volume decomposition metrics
    """
    
    def __init__(self, paths_df: pd.DataFrame):
        """Initialize analytics with paths data."""
        self.paths_df = paths_df
        self.endpoint_df = self._calculate_endpoints()
    
    def _calculate_endpoints(self) -> pd.DataFrame:
        """Calculate endpoint statistics for all paths."""
        endpoints = []
        
        for j in range(len(self.paths_df)):
            if j >= len(self.paths_df):
                break
                
            path_data = self.paths_df.iloc[j, 1]
            
            # Get last values
            last_row = path_data.iloc[-1]
            cum_II = 1 + last_row['interest_income']
            cum_II_netto = 1 + last_row['II_netto']
            
            # Create endpoint row
            endpoint_row = last_row.to_dict()
            endpoint_row.update({
                'cum_II': cum_II,
                'cum_II_netto': cum_II_netto,
                'path_id': j
            })
            
            # Add volume decomposition metrics if available
            vol_cols = [col for col in path_data.columns if 'vol' in col]
            for col in vol_cols:
                endpoint_row[col] = path_data.iloc[-1][col]
            
            endpoints.append(endpoint_row)
        
        return pd.DataFrame(endpoints)
    
    def endpoint_stat(self) -> pd.DataFrame:
        """Calculate descriptive statistics for endpoints."""
        if self.endpoint_df.empty:
            return pd.DataFrame()
        
        # Standard statistics columns
        stats_cols = ['cum_II', 'cum_II_netto', 'FX']
        
        # Add volume decomposition columns if available
        vol_cols = [col for col in self.endpoint_df.columns 
                   if 'vol' in col and ('V_total' in col or 'V_required' in col or 'V_excess' in col)]
        stats_cols.extend(vol_cols)
        
        available_cols = [col for col in stats_cols if col in self.endpoint_df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        return self.endpoint_df[available_cols].describe()
    
    def volume_decomposition_stat(self) -> pd.DataFrame:
        """Calculate detailed statistics for volume decomposition."""
        if self.endpoint_df.empty:
            return pd.DataFrame()
        
        # Filter volume decomposition columns
        vol_cols = [col for col in self.endpoint_df.columns if 'vol' in col]
        
        if not vol_cols:
            return pd.DataFrame()
        
        vol_df = self.endpoint_df[vol_cols]
        
        # Calculate additional statistics
        stats = vol_df.describe()
        
        # Add ratio statistics
        if 'V_total_vol' in vol_df.columns and 'V_required_vol' in vol_df.columns:
            total_sum = vol_df['V_total_vol'].sum()
            required_sum = vol_df['V_required_vol'].sum()
            excess_sum = vol_df['V_excess_vol'].sum() if 'V_excess_vol' in vol_df.columns else 0
            
            ratio_stats = pd.DataFrame({
                'V_required/V_total': [required_sum / max(total_sum, 1e-10)],
                'V_excess/V_total': [excess_sum / max(total_sum, 1e-10)],
                'Total V_required': [required_sum],
                'Total V_excess': [excess_sum],
                'Mean sufficiency ratio': [vol_df['V_total_vol'].mean() / max(vol_df['V_required_vol'].mean(), 1e-10)]
            })
            
            stats = pd.concat([stats, ratio_stats], axis=0)
        
        return stats


class Visualization:
    """
    Optimized visualization class with volume decomposition support.
    
    Changes:
    1. Reduced redundant calculations
    2. Better plotting efficiency
    3. Added volume decomposition visualizations
    """
    
    def __init__(self, paths_df: pd.DataFrame, pool_performance: pd.DataFrame,
                 dropped_paths_log: List[Dict] = None):
        """Initialize visualization."""
        self.paths_df = paths_df
        self.pool_performance = pool_performance
        self.dropped_paths_log = dropped_paths_log or []
        self.analytics = Analytics(paths_df)
        self.amm_engine = Payoff()
        
        # Calculate endpoint statistics
        if not self.analytics.endpoint_df.empty:
            self.endpoint_x_range = [
                np.floor(self.analytics.endpoint_df['FX'].min() * 100) / 100,
                np.ceil(self.analytics.endpoint_df['FX'].max() * 100) / 100
            ]
            self.endpoint_y_mean = self.analytics.endpoint_df['cum_II_netto'].mean()
            self.endpoint_y_std = self.analytics.endpoint_df['cum_II_netto'].std(ddof=1)
        else:
            self.endpoint_x_range = [0, 1]
            self.endpoint_y_mean = 0
            self.endpoint_y_std = 0
    
    def path_plot(self, show_volume_decomposition: bool = False) -> None:
        """
        Create 3D plot of simulation paths.
        
        Args:
            show_volume_decomposition: If True, color paths by volume sufficiency
        """
        fig = plt.figure(figsize=(18.5, 10.5))
        ax = fig.add_subplot(111, projection='3d')
        
        y = list(self.pool_performance.index)
        
        # Limit number of paths for better performance
        max_paths_to_plot = min(100, len(self.paths_df))
        
        for j in range(max_paths_to_plot):
            if j >= len(self.paths_df):
                break
                
            path_data = self.paths_df.iloc[j, 1]
            x = path_data['FX'].values
            z = 1 + path_data['II_netto'].values
            
            # Determine color based on volume decomposition if requested
            if show_volume_decomposition and 'V_total_vol' in path_data.columns:
                # Calculate average volume sufficiency ratio for this path
                v_total = path_data['V_total_vol'].mean()
                v_required = path_data['V_required_vol'].mean()
                if v_required > 0:
                    sufficiency_ratio = v_total / v_required
                    # Color code: green for sufficient, red for insufficient
                    if sufficiency_ratio >= 1:
                        color = 'green'
                        alpha = 0.3
                    else:
                        color = 'red'
                        alpha = 0.6
                else:
                    color = 'blue'
                    alpha = 0.5
            else:
                color = 'blue'
                alpha = 0.5
            
            # Plot path
            ax.plot(x, y[:len(x)], z, alpha=alpha, linewidth=0.5, color=color)
        
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Days')
        ax.set_zlabel('Returns')
        
        if show_volume_decomposition:
            ax.set_title('Simulation Paths - Colored by Volume Sufficiency\n'
                        'Green: Sufficient volume, Red: Insufficient volume')
        else:
            ax.set_title('Simulation Paths - 3D View')
        
        plt.show()
    
    def endpoint_plot(self, IL_curve_x_range: List[float] = None,
                     IL_curve_y_offset: List[float] = None,
                     show_volume_size: bool = False) -> None:
        """
        Create scatter plot of endpoints with IL curve.
        
        Args:
            IL_curve_x_range: Range for IL curve calculation
            IL_curve_y_offset: Vertical offset for IL curve
            show_volume_size: If True, size points by total volume
        """
        if self.analytics.endpoint_df.empty:
            print("No endpoint data available")
            return
        
        # Get initial FX
        FX_t0 = self.paths_df.iloc[0, 1]['FX'][0] if len(self.paths_df) > 0 else 1.0
        
        # Set IL curve parameters
        if IL_curve_x_range is None:
            IL_curve_start, IL_curve_end = self.endpoint_x_range
        else:
            IL_curve_start, IL_curve_end = IL_curve_x_range
        
        if IL_curve_y_offset is None:
            peak = self.endpoint_y_mean + 3 * self.endpoint_y_std
        else:
            peak = IL_curve_y_offset[0]
        
        # Create IL curve
        n_points = 100
        IL_curve_FX = np.linspace(IL_curve_start, IL_curve_end, n_points)
        IL_curve_returns = np.array([
            self.amm_engine.impermanent_loss(FX_t0, fx, relative='Y') + peak
            for fx in IL_curve_FX
        ])
        
        # Create plot
        fig = plt.figure(figsize=(18.5, 10.5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare scatter plot data
        endpoint_day = len(self.pool_performance) - 1
        
        # Determine point sizes if showing volume size
        if show_volume_size and 'V_total_vol' in self.analytics.endpoint_df.columns:
            volumes = self.analytics.endpoint_df['V_total_vol'].values
            sizes = 10 + 100 * (volumes / max(volumes.max(), 1))
            color = volumes
            cmap = 'viridis'
        else:
            sizes = 10
            color = 'blue'
            cmap = None
        
        # Plot endpoints
        scatter = ax.scatter(
            self.analytics.endpoint_df['FX'],
            np.full(len(self.analytics.endpoint_df), endpoint_day),
            self.analytics.endpoint_df['cum_II_netto'],
            alpha=0.5,
            s=sizes,
            c=color,
            cmap=cmap
        )
        
        if show_volume_size and cmap:
            plt.colorbar(scatter, ax=ax, label='Total Volume')
        
        # Plot IL curve
        ax.plot(
            IL_curve_FX,
            np.full(n_points, endpoint_day),
            IL_curve_returns,
            'r-',
            linewidth=3,
            label='IL Curve'
        )
        
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Days')
        ax.set_zlabel('Returns')
        
        title = 'Endpoint Distribution with IL Curve'
        if show_volume_size:
            title += ' (Point size proportional to volume)'
        ax.set_title(title)
        
        ax.legend()
        plt.show()
    
    def volume_decomposition_plot(self) -> None:
        """Create visualization of volume decomposition analysis."""
        if self.analytics.endpoint_df.empty:
            print("No endpoint data available")
            return
        
        # Check if volume decomposition columns exist
        vol_cols = [col for col in self.analytics.endpoint_df.columns 
                   if 'vol' in col and ('V_total' in col or 'V_required' in col)]
        
        if not vol_cols:
            print("No volume decomposition data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Volume decomposition by path
        if 'V_total_vol' in self.analytics.endpoint_df.columns and 'V_required_vol' in self.analytics.endpoint_df.columns:
            paths = range(len(self.analytics.endpoint_df))
            axes[0].bar(paths, self.analytics.endpoint_df['V_total_vol'], 
                       alpha=0.7, label='Total Volume', color='blue')
            axes[0].bar(paths, self.analytics.endpoint_df['V_required_vol'], 
                       alpha=0.7, label='Required Volume', color='orange')
            axes[0].set_xlabel('Path ID')
            axes[0].set_ylabel('Volume')
            axes[0].set_title('Volume Decomposition by Path')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Volume sufficiency ratio
        if 'V_total_vol' in self.analytics.endpoint_df.columns and 'V_required_vol' in self.analytics.endpoint_df.columns:
            sufficiency_ratio = self.analytics.endpoint_df['V_total_vol'] / \
                               np.maximum(self.analytics.endpoint_df['V_required_vol'], 1e-10)
            axes[1].hist(sufficiency_ratio, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Sufficiency Threshold')
            axes[1].set_xlabel('Volume Sufficiency Ratio (Total/Required)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Volume Sufficiency Ratios')
            axes[1].legend()
        
        # Plot 3: Returns vs Volume Sufficiency
        if 'V_total_vol' in self.analytics.endpoint_df.columns and 'V_required_vol' in self.analytics.endpoint_df.columns:
            sufficiency_ratio = self.analytics.endpoint_df['V_total_vol'] / \
                               np.maximum(self.analytics.endpoint_df['V_required_vol'], 1e-10)
            returns = self.analytics.endpoint_df['cum_II_netto']
            scatter = axes[2].scatter(sufficiency_ratio, returns, 
                                     c=self.analytics.endpoint_df['FX'], 
                                     cmap='viridis', alpha=0.6)
            axes[2].axvline(x=1, color='red', linestyle='--', linewidth=2, 
                           label='Sufficiency Threshold')
            axes[2].set_xlabel('Volume Sufficiency Ratio')
            axes[2].set_ylabel('Cumulative Returns')
            axes[2].set_title('Returns vs Volume Sufficiency Ratio')
            axes[2].legend()
            plt.colorbar(scatter, ax=axes[2], label='Final Price')
        
        # Plot 4: Dropped paths analysis (if any)
        if self.dropped_paths_log:
            dropped_counts = {}
            for log in self.dropped_paths_log:
                path_id = log['path']
                if path_id not in dropped_counts:
                    dropped_counts[path_id] = 0
                dropped_counts[path_id] += 1
            
            axes[3].bar(dropped_counts.keys(), dropped_counts.values(), 
                       color='red', alpha=0.7)
            axes[3].set_xlabel('Path ID')
            axes[3].set_ylabel('Number of Dropped Steps')
            axes[3].set_title('Distribution of Dropped Path Steps')
            axes[3].tick_params(axis='x', rotation=45)
        else:
            axes[3].text(0.5, 0.5, 'No paths were dropped\n(All had sufficient volume)',
                        ha='center', va='center', fontsize=12)
            axes[3].set_title('Dropped Paths Analysis')
        
        plt.tight_layout()
        plt.show()
    
    def summary_statistics(self) -> None:
        """Print summary statistics including volume decomposition."""
        print("=" * 60)
        print("SIMULATION SUMMARY STATISTICS")
        print("=" * 60)
        
        # Basic endpoint statistics
        if not self.analytics.endpoint_df.empty:
            basic_stats = self.analytics.endpoint_stat()
            print("\n1. Endpoint Statistics:")
            print(basic_stats)
        
        # Volume decomposition statistics
        vol_stats = self.analytics.volume_decomposition_stat()
        if not vol_stats.empty:
            print("\n2. Volume Decomposition Statistics:")
            print(vol_stats)
        
        # Dropped paths information
        if self.dropped_paths_log:
            print(f"\n3. Dropped Paths: {len(self.dropped_paths_log)} steps were dropped")
            print("   First 5 dropped steps:")
            for i, log in enumerate(self.dropped_paths_log[:5]):
                print(f"   Step {i+1}: Path {log['path']}, Time {log['time_step']}, "
                      f"Deficit: {log['deficit']:.2f}")
            if len(self.dropped_paths_log) > 5:
                print(f"   ... and {len(self.dropped_paths_log) - 5} more")
        else:
            print("\n3. No paths were dropped (all had sufficient volume)")
        
        print("=" * 60)


class OnePipeline:
    """
    Optimized single pipeline class with volume decomposition support.
    
    Changes:
    1. Better parameter handling
    2. Error checking
    3. More efficient simulation
    4. Volume decomposition integration
    """
    
    def __init__(self, pool_fee: float, amount_x_pool_t0: float, 
                amount_y_pool_t0: float, total_investment_x: float,
                max_paths: int, deposit_split_percentage: float,
                predicted_period: int, volume: pd.Series,
                theta: float,
                captured_volume_perc: float = 1.0,
                FX_data: pd.DataFrame = pd.DataFrame(),
                ticker: str = '', start_date: List[int] = None,
                end_date: Union[List[int], str] = None,
                interval: str = '', backtesting: bool = True,
                plot_sim: bool = False,
                use_volume_decomposition: bool = False,
                drop_insufficient_volume: bool = False,
                show_volume_plots: bool = True):
        """Initialize and run complete pipeline."""
        
        # Validate inputs
        self._validate_inputs(
            pool_fee, amount_x_pool_t0, amount_y_pool_t0,
            total_investment_x, max_paths, deposit_split_percentage
        )
        
        # Store configuration
        self.use_volume_decomposition = use_volume_decomposition
        self.drop_insufficient_volume = drop_insufficient_volume
        self.show_volume_plots = show_volume_plots
        
        # Initialize components
        self.price = PriceSim()
        self.volume_sim = VolumeSim()
        self.amm = Payoff()
        
        # Run price simulation
        print("=" * 60)
        print("STEP 1: Running price simulation...")
        print("=" * 60)
        self.price.pipeline(
            predicted_period=predicted_period,
            FX_data=FX_data,
            ticker=ticker,
            start_date=start_date or [2013, 8, 12],
            end_date=end_date or 'today',
            interval=interval or '1d',
            backtesting=backtesting,
            plot_sim=plot_sim
        )
        
        # Run volume simulation
        print("\n" + "=" * 60)
        print("STEP 2: Running volume simulation...")
        print("=" * 60)
        volume_sim_result = self.volume_sim.pipeline(
            volume=volume,
            theta=theta,
            predicted_period=predicted_period,
            backtesting=backtesting,
            plot_sim=plot_sim  # Add this if you want plotting
        )
        
        # Adjust volume by captured percentage
        adjusted_volume = volume_sim_result * captured_volume_perc
        
        # Run AMM payoff calculation with volume decomposition
        print("\n" + "=" * 60)
        print("STEP 3: Running AMM payoff calculation...")
        print("=" * 60)
        if use_volume_decomposition:
            print(f"Using volume decomposition: YES")
            print(f"Dropping insufficient volume paths: {drop_insufficient_volume}")
        else:
            print("Using volume decomposition: NO")
        
        self.amm.pipeline(
            pool_fee=pool_fee,
            amount_x_pool_t0=amount_x_pool_t0,
            amount_y_pool_t0=amount_y_pool_t0,
            total_investment_x=total_investment_x,
            FX_timeseries=self.price.sim,
            volume_timeseries=adjusted_volume,
            max_paths=max_paths,
            deposit_split_percentage=deposit_split_percentage,
            use_volume_decomposition=use_volume_decomposition,
            drop_insufficient_volume=drop_insufficient_volume
        )
        
        # Create visualizations
        print("\n" + "=" * 60)
        print("STEP 4: Creating visualizations...")
        print("=" * 60)
        
        self.plot = Visualization(
            paths_df=self.amm.paths_df,
            pool_performance=self.amm.pool_performance,
            dropped_paths_log=self.amm.dropped_paths_log
        )
        
        # Standard plots
        print("\n1. Standard 3D path plot...")
        self.plot.path_plot(show_volume_decomposition=use_volume_decomposition)
        
        print("\n2. Endpoint distribution plot...")
        self.plot.endpoint_plot(show_volume_size=use_volume_decomposition)
        
        # Volume decomposition plots if enabled
        if use_volume_decomposition and show_volume_plots:
            print("\n3. Volume decomposition analysis...")
            self.plot.volume_decomposition_plot()
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        self.plot.summary_statistics()
        
        print("\nPipeline completed successfully!")
    
    def _validate_inputs(self, pool_fee: float, amount_x_pool_t0: float,
                        amount_y_pool_t0: float, total_investment_x: float,
                        max_paths: int, deposit_split_percentage: float) -> None:
        """Validate pipeline inputs."""
        if pool_fee < 0 or pool_fee > 1:
            raise ValueError("pool_fee must be between 0 and 1")
        
        if amount_x_pool_t0 <= 0 or amount_y_pool_t0 <= 0:
            raise ValueError("Pool amounts must be positive")
        
        if total_investment_x <= 0:
            raise ValueError("Total investment must be positive")
        
        if max_paths <= 0:
            raise ValueError("max_paths must be positive")
        
        if deposit_split_percentage < 0 or deposit_split_percentage > 1:
            raise ValueError("deposit_split_percentage must be between 0 and 1")
    
    def get_volume_decomposition_summary(self) -> pd.DataFrame:
        """Get volume decomposition summary from AMM."""
        if hasattr(self.amm, 'get_volume_decomposition_summary'):
            return self.amm.get_volume_decomposition_summary()
        return pd.DataFrame()
    
    def export_results(self, output_dir: str = ".") -> None:
        """Export simulation results to CSV files."""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export paths data
        if not self.amm.paths_df.empty:
            paths_file = os.path.join(output_dir, "amm_simulation_paths.csv")
            # Extract all path data into a single DataFrame
            all_paths = []
            for i in range(len(self.amm.paths_df)):
                path_data = self.amm.paths_df.iloc[i, 1].copy()
                path_data['path_id'] = i
                all_paths.append(path_data)
            
            if all_paths:
                combined_df = pd.concat(all_paths, ignore_index=True)
                combined_df.to_csv(paths_file, index=False)
                print(f"Exported paths data to: {paths_file}")
        
        # Export endpoint statistics
        if not self.plot.analytics.endpoint_df.empty:
            endpoints_file = os.path.join(output_dir, "endpoint_statistics.csv")
            self.plot.analytics.endpoint_df.to_csv(endpoints_file, index=False)
            print(f"Exported endpoint statistics to: {endpoints_file}")
        
        # Export dropped paths log if available
        if self.amm.dropped_paths_log:
            dropped_file = os.path.join(output_dir, "dropped_paths_log.csv")
            dropped_df = pd.DataFrame(self.amm.dropped_paths_log)
            dropped_df.to_csv(dropped_file, index=False)
            print(f"Exported dropped paths log to: {dropped_file}")
        
        print(f"\nAll results exported to: {os.path.abspath(output_dir)}")


# ========== Enhanced Main Execution Example ==========
if __name__ == "__main__":
    """
    Example usage of the enhanced AMM calculator with volume decomposition.
    
    Note: This requires actual data to run properly.
    """
    
    # Example parameters without volume decomposition
    example_params_standard = {
        'pool_fee': 0.01,
        'amount_x_pool_t0': 1000000.0,
        'amount_y_pool_t0': 1000000.0,
        'total_investment_x': 10000.0,
        'max_paths': 50,
        'deposit_split_percentage': 0.5,
        'predicted_period': 260,
        'captured_volume_perc': 1.0,
        'backtesting': True,
        'plot_sim': False,
        'use_volume_decomposition': False,  # Standard mode
        'drop_insufficient_volume': False,
        'show_volume_plots': False
    }
    
    # Example parameters with volume decomposition
    example_params_decomposition = {
        'pool_fee': 0.01,
        'amount_x_pool_t0': 1000000.0,
        'amount_y_pool_t0': 1000000.0,
        'total_investment_x': 10000.0,
        'max_paths': 100,
        'deposit_split_percentage': 0.5,
        'predicted_period': 260,
        'captured_volume_perc': 0.8,
        'backtesting': True,
        'plot_sim': False,
        'use_volume_decomposition': True,  # Enhanced mode
        'drop_insufficient_volume': True,  # Drop paths with insufficient volume
        'show_volume_plots': True
    }
    
    print("ENHANCED AMM CALCULATOR WITH VOLUME DECOMPOSITION")
    print("=" * 60)
    print("\nAvailable operation modes:")
    print("1. Standard mode - Traditional payoff calculation")
    print("2. Enhanced mode - With volume decomposition analysis")
    print("\nKey features of enhanced mode:")
    print("  • Calculates minimum required volume for price moves")
    print("  • Decomposes volume into arbitrage and noise components")
    print("  • Directional fee allocation based on arbitrage flow")
    print("  • Optional filtering of paths with insufficient volume")
    print("  • Comprehensive visualization of volume dynamics")
    
    print("\nAvailable classes:")
    print("1. PriceSim - Price simulations using GBM")
    print("2. VolumeSim - Volume simulations using UOP")
    print("3. Payoff - Enhanced AMM payoff calculations")
    print("4. Analytics - Statistical analysis with volume metrics")
    print("5. Visualization - Enhanced plotting with volume analysis")
    print("6. OnePipeline - Complete pipeline with decomposition support")
    
    # Note: To run the pipeline, you need actual data
    # Example usage:
    # 
    # # Standard pipeline
    # pipeline_std = OnePipeline(**example_params_standard, volume=your_volume_data)
    #
    # # Enhanced pipeline with volume decomposition
    # pipeline_enh = OnePipeline(**example_params_decomposition, volume=your_volume_data)
    #
    # # Export results
    # pipeline_enh.export_results("simulation_results")
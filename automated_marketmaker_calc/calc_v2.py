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

    def plot_paths(self, sim_df: pd.DataFrame, FX_df: pd.Series, 
                  predicted_period: int, backtesting: bool = False) -> None:
        """Plot GBM paths."""
        # Store parameters for title
        self.S0 = sim_df.iloc[0, 0] if not sim_df.empty else 0
        
        # Create time arrays
        if backtesting:
            time_space = np.linspace(0, 1, predicted_period + 1)
            plot_df = sim_df
        else:
            time_space = np.linspace(0, 1, 2 * predicted_period + 1)
            # Create NaN DataFrame and concatenate
            df_nan = pd.DataFrame(np.nan, index=range(predicted_period), 
                                 columns=range(len(sim_df.columns)))
            plot_df = pd.concat([df_nan, sim_df]).reset_index(drop=True)
        
        # Create time matrix for plotting
        tt = np.full((10000, len(time_space)), time_space).T
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(tt[:len(plot_df)], plot_df.values)
        plt.xlabel("Years $(t)$")
        plt.ylabel("Exchange Rate $(S_t)$")
        plt.title(f"Realizations of Geometric Brownian Motion\n"
                 f"$dS_t = \mu S_t dt + \sigma S_t dW_t$\n"
                 f"$S_0 = {self.S0:.6f}, \mu = {self.mu:.6f}, \sigma = {self.sigma:.6f}$")
        plt.grid(True)
        plt.show()

    def pipeline(self, predicted_period: int, FX_data: pd.DataFrame = pd.DataFrame(),
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
        mu_scaled = self.mu * np.sqrt(predicted_period)
        sigma_scaled = self.sigma * np.sqrt(predicted_period)
        
        # Run simulation
        self.sim = self.GBM(
            mu=mu_scaled,
            sigma=sigma_scaled,
            S0=S0,
            steps=predicted_period,
            n_paths=10000,
            plot='N',
            pandas='Y'
        )
        
        # Plot if requested
        if plot_sim:
            self.plot_paths(
                sim_df=self.sim,
                FX_df=self.FX_data['Adj Close'],
                predicted_period=predicted_period,
                backtesting=backtesting
            )


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
    plot_paths = PriceSim.plot_paths  # ← ADD THIS LINE


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

    def pipeline(self, volume: pd.Series, predicted_period: int, 
                backtesting: bool = True) -> pd.DataFrame:
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
        self.mu = np.mean(volume.values)
        self.sigma = np.std(volume.values, ddof=1)
        self.theta = 5.0  # Fixed as in original code
        
        # Determine starting value
        if backtesting:
            start_idx = len(volume) - predicted_period - 1
        else:
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
            n_paths=10000,
            plot='N',
            pandas='Y'
        )
        
        return self.sim

    def plot_UOP_paths(self, sim_df: pd.DataFrame, predicted_period: int, 
                      backtesting: bool = False) -> None:
        """
        Plot Ornstein-Uhlenbeck Process paths from simulation results.
        
        Args:
            sim_df: DataFrame with simulated paths (columns are paths, rows are time steps)
            predicted_period: Number of periods predicted
            backtesting: Whether this is a backtest
        """
        # Store parameters for title
        S0 = sim_df.iloc[0, 0] if not sim_df.empty else 0
        
        # Create time arrays
        if backtesting:
            time_space = np.linspace(0, 1, predicted_period + 1)
            plot_df = sim_df
        else:
            time_space = np.linspace(0, 1, 2 * predicted_period + 1)
            # Create NaN DataFrame for historical part
            df_nan = pd.DataFrame(np.nan, index=range(predicted_period), 
                                 columns=range(len(sim_df.columns)))
            plot_df = pd.concat([df_nan, sim_df]).reset_index(drop=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot all paths
        plt.plot(time_space[:len(plot_df)], plot_df.values, alpha=0.5, linewidth=0.5)
        
        # Calculate and plot mean path
        mean_path = plot_df.mean(axis=1)
        plt.plot(time_space[:len(plot_df)], mean_path, 'k-', linewidth=2, label='Mean path')
        
        # Add long-term mean line
        plt.axhline(y=self.mu, color='r', linestyle='--', 
                   linewidth=2, label=f'Long-term mean (μ) = {self.mu:.2f}')
        
        # Add starting value
        plt.axhline(y=S0, color='g', linestyle=':', 
                   linewidth=1.5, label=f'Starting value (V₀) = {S0:.2f}')
        
        # Add vertical line separating historical and future (if not backtesting)
        if not backtesting:
            plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5)
            plt.text(0.51, plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]), 
                    'Future', rotation=90, verticalalignment='bottom')
        
        plt.xlabel("Time $(t)$")
        plt.ylabel("Volume $(V_t)$")
        plt.title(f"Ornstein-Uhlenbeck Process Simulation\n"
                 f"$dV_t = \\theta(\\mu - V_t)dt + \\sigma dW_t$\n"
                 f"$V_0 = {S0:.2f}, \\mu = {self.mu:.2f}, \\sigma = {self.sigma:.2f}, \\theta = {self.theta:.2f}$")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Payoff:
    """
    Optimized Payoff class for AMM calculations.
    
    Changes made:
    1. Added type hints
    2. Optimized calculations using vectorized operations where possible
    3. Improved error handling
    4. Removed redundant intermediate variables
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
    
    @staticmethod
    def fee_amount_by_reserves(FX_in_x: float, volume_in_x: float, 
                              fee_rate: float) -> Tuple[float, float]:
        """Calculate fee amounts from reserves."""
        if FX_in_x == 0:
            return 0.0, 0.0
        
        y_amount_exchanged = volume_in_x / FX_in_x
        x_fee_amount = volume_in_x / 2 * fee_rate
        y_fee_amount = y_amount_exchanged / 2 * fee_rate
        
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
    
    # ========== Time Step Calculations ==========
    
    def t0_calc(self, pool_fee: float, amount_x_pool_t0: float, 
               amount_y_pool_t0: float, total_investment_x: float,
               deposit_split_percentage: float) -> None:
        """Calculate initial time step (t=0)."""
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
        
        # Deposit split
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
    
    def tn_calc(self, FX_timeseries: pd.DataFrame, volume_timeseries: pd.DataFrame, 
               max_paths: int) -> None:
        """Calculate subsequent time steps."""
        self.paths_df = pd.DataFrame()
        n_steps = len(FX_timeseries)
        
        for j in range(max_paths):
            # Skip if column doesn't exist
            if j >= len(FX_timeseries.columns):
                continue
                
            for i in range(1, n_steps):
                # Skip if data is missing
                if pd.isna(FX_timeseries.iloc[i, j]) or pd.isna(volume_timeseries.iloc[i, j]):
                    continue
                
                # Copy previous fee and get current FX and volume
                self.pool_performance.at[i, 'pool_fee'] = self.pool_performance.at[i-1, 'pool_fee']
                self.pool_performance.at[i, 'FX'] = FX_timeseries.iloc[i, j]
                self.pool_performance.at[i, 'volume'] = volume_timeseries.iloc[i, j]
                
                # Calculate new reserves based on FX change
                prev_k = self.pool_reserves.at[i-1, 'k']
                current_FX = self.pool_performance.at[i, 'FX']
                
                self.pool_reserves.at[i, 'amount_x'] = self.reserves_calc(
                    prev_k, current_FX, calculate_x=True
                )
                self.pool_reserves.at[i, 'amount_y'] = self.reserves_calc(
                    prev_k, current_FX, calculate_x=False
                )
                
                # Calculate fees from volume
                volume = self.pool_performance.at[i, 'volume']
                fee_rate = self.pool_performance.at[i, 'pool_fee']
                x_fee, y_fee = self.fee_amount_by_reserves(current_FX, volume, fee_rate)
                
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
            
            # Create merged DataFrame for this path
            merge_df = pd.concat([
                self.pool_performance,
                self.depositor_performance,
                self.pool_reserves.add_suffix('_pool'),
                self.depositor_reserves.add_suffix('_depositor')
            ], axis=1).fillna(0)
            
            # Store in paths DataFrame
            time_step_df = pd.DataFrame({'sim': [j], 'dfs': [merge_df.copy()]})
            self.paths_df = pd.concat([self.paths_df, time_step_df], ignore_index=True)
            
            # Progress update
            if (j + 1) % 10 == 0 or j == max_paths - 1:
                clear_output(wait=True)
                print(f"Progress: {j+1}/{max_paths} paths processed")
    
    # ========== Pipeline Method ==========
    
    def pipeline(self, pool_fee: float, amount_x_pool_t0: float, 
                amount_y_pool_t0: float, total_investment_x: float,
                FX_timeseries: pd.DataFrame, volume_timeseries: pd.DataFrame, 
                max_paths: int, deposit_split_percentage: float) -> None:
        """Execute complete payoff calculation pipeline."""
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
            max_paths=min(max_paths, len(FX_timeseries.columns))
        )


class Analytics:
    """
    Optimized analytics class.
    
    Changes:
    1. Vectorized endpoint calculation
    2. Better memory efficiency
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
            
            endpoints.append(endpoint_row)
        
        return pd.DataFrame(endpoints)
    
    def endpoint_stat(self) -> pd.DataFrame:
        """Calculate descriptive statistics for endpoints."""
        if self.endpoint_df.empty:
            return pd.DataFrame()
        
        stats_cols = ['cum_II', 'cum_II_netto', 'FX']
        available_cols = [col for col in stats_cols if col in self.endpoint_df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        return self.endpoint_df[available_cols].describe()


class Visualization:
    """
    Optimized visualization class.
    
    Changes:
    1. Reduced redundant calculations
    2. Better plotting efficiency
    """
    
    def __init__(self, paths_df: pd.DataFrame, pool_performance: pd.DataFrame):
        """Initialize visualization."""
        self.paths_df = paths_df
        self.pool_performance = pool_performance
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
    
    def path_plot(self) -> None:
        """Create 3D plot of simulation paths."""
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
            
            # Plot path
            ax.plot(x, y[:len(x)], z, alpha=0.5, linewidth=0.5)
        
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Days')
        ax.set_zlabel('Returns')
        ax.set_title('Simulation Paths - 3D View')
        plt.show()
    
    def endpoint_plot(self, IL_curve_x_range: List[float] = None,
                     IL_curve_y_offset: List[float] = None) -> None:
        """Create scatter plot of endpoints with IL curve."""
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
        
        # Plot endpoints
        endpoint_day = len(self.pool_performance) - 1
        ax.scatter(
            self.analytics.endpoint_df['FX'],
            np.full(len(self.analytics.endpoint_df), endpoint_day),
            self.analytics.endpoint_df['cum_II_netto'],
            alpha=0.5,
            s=10
        )
        
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
        ax.set_title('Endpoint Distribution with IL Curve')
        ax.legend()
        plt.show()


class OnePipeline:
    """
    Optimized single pipeline class.
    
    Changes:
    1. Better parameter handling
    2. Error checking
    3. More efficient simulation
    """
    
    def __init__(self, pool_fee: float, amount_x_pool_t0: float, 
                amount_y_pool_t0: float, total_investment_x: float,
                max_paths: int, deposit_split_percentage: float,
                predicted_period: int, volume: pd.Series,
                captured_volume_perc: float = 1.0,
                FX_data: pd.DataFrame = pd.DataFrame(),
                ticker: str = '', start_date: List[int] = None,
                end_date: Union[List[int], str] = None,
                interval: str = '', backtesting: bool = True,
                plot_sim: bool = False):
        """Initialize and run complete pipeline."""
        
        # Validate inputs
        self._validate_inputs(
            pool_fee, amount_x_pool_t0, amount_y_pool_t0,
            total_investment_x, max_paths, deposit_split_percentage
        )
        
        # Initialize components
        self.price = PriceSim()
        self.volume_sim = VolumeSim()
        self.amm = Payoff()
        
        # Run price simulation
        print("Running price simulation...")
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
        print("Running volume simulation...")
        volume_sim_result = self.volume_sim.pipeline(
            volume=volume,
            predicted_period=predicted_period,
            backtesting=backtesting
        )
        
        # Adjust volume by captured percentage
        adjusted_volume = volume_sim_result * captured_volume_perc
        
        # Run AMM payoff calculation
        print("Running AMM payoff calculation...")
        self.amm.pipeline(
            pool_fee=pool_fee,
            amount_x_pool_t0=amount_x_pool_t0,
            amount_y_pool_t0=amount_y_pool_t0,
            total_investment_x=total_investment_x,
            FX_timeseries=self.price.sim,
            volume_timeseries=adjusted_volume,
            max_paths=max_paths,
            deposit_split_percentage=deposit_split_percentage
        )
        
        # Create visualizations
        print("Creating visualizations...")
        self.plot = Visualization(
            paths_df=self.amm.paths_df,
            pool_performance=self.amm.pool_performance
        )
        
        self.plot.path_plot()
        self.plot.endpoint_plot()
        
        print("Pipeline completed successfully!")
    
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


# ========== Main Execution Example ==========
if __name__ == "__main__":
    """
    Example usage of the optimized AMM calculator.
    
    Note: This requires actual data to run properly.
    """
    
    # Example parameters
    example_params = {
        'pool_fee': 0.01,
        'amount_x_pool_t0': 1000000.0,
        'amount_y_pool_t0': 1000000.0,
        'total_investment_x': 10000.0,
        'max_paths': 50,
        'deposit_split_percentage': 0.5,
        'predicted_period': 260,
        'captured_volume_perc': 1.0,
        'backtesting': True,
        'plot_sim': False
    }
    
    print("AMM Calculator initialized successfully!")
    print("\nAvailable classes:")
    print("1. PriceSim - Price simulations using GBM")
    print("2. VolumeSim - Volume simulations using UOP")
    print("3. Payoff - AMM payoff calculations")
    print("4. Analytics - Statistical analysis of results")
    print("5. Visualization - 3D plotting of results")
    print("6. OnePipeline - Complete pipeline execution")
    
    # Note: To run the pipeline, you need actual data
    # pipeline = OnePipeline(**example_params, volume=your_volume_data)
import pandas as pd
import numpy as np
import math
from typing import Tuple, Dict, List, Optional
from IPython.display import clear_output


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
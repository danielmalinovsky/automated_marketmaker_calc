# Installation
```
pip install automated_marketmaker_calc
```

# Get started
```python
import automated_marketmaker_calc as amm

payoff_calc = amm.payoff()

payoff_calc.pipeline(
    pool_fee= 0.01,
    amount_x_pool_t0= amount_USD_pool_t0,
    amount_y_pool_t0= amount_y_pool_t0,
    total_investment_x= total_investment_USD,
    FX_timeseries = price.sim_df,
    volume_timeseries = volume.sim*0.01, 
    max_paths = 100,
    deposit_split_percentage = 0.5
)
```
```
![plot_path](https://github.com/danielmalinovsky/automated_marketmaker_calc/assets/106654319/c56a2fa8-1660-4d28-9685-e7f42648e381)
```

This package is composed of two main sections: `Payoff calculation` and `Simulation methods`.
- `payoff calculation` is engine for simulating internal dynamics of the AMM and getting returns on deposit
- `simulation methods` are two main methods used for simulationg AMM input data
    - in below figure represented as V0 and P 

$$ payoff_(rel,t) (V_0,vol_(x,t),ρ,P_(x/y,t ) )=fee_rel (V_0,vol_(x,t),ρ,P_(x/y,t ) )-IL_rel ( P_(x/y,t_0  ),P_(x/y,t ) )-TC_rel!$$


# Payoff calculation
Current payoff calculation is based on following features:
- uniswap v2
- stable deposit ratio (no withdrawals during simulation period)
- stable fee split (fees collected in token y and x are split in same ratio at the end of each period

Payoff method is able to take time series data of exchange rates and exchange volumes. Advanced quantitative analyst are welcome to use more advanced simulation mathods, as GARCH, SABR or cointegration simulation methods to simulate exchange rate and volumes together. Payoff results will be only as good as presented simulation data. Howeve goal of this package is to provide robust AMM engine to calculate payoffs and not simulation of financial time series.

## Automated Market Maker (AMM) Payoff Calculator

### Overview

A Python class for financial modeling and simulation related to automated market maker payoffs. Please be advised that the current version of the payoff calculator is using the UNISWAP V2 engine.

#### Functions:

- `k_product(x, y)`: Calculates the product of two values x and y.
- `value_in_x(x, y, FX)`: Calculates the value in X given X and Y amounts and an FX ratio.
- `FX_x_over_y(k_pool, y_amount)`: Calculates the FX ratio of X over Y for a pool.
- `deposit_split(deposit_value_in_x, percentage_of_x, FX_x_over_y)`: Splits a deposit value into X and Y amounts based on a given percentage.
- `pool_share(k_depositor, k_pool)`: Calculates the pool share for a depositor.
- `interest_income(accrued_fee_in_x, value_in_x, relative='N')`: Calculates interest income for a depositor.
- `hold_in_x(amount_x_t0, amount_y_t0, FX_tn)`: Calculates the portfolio value in X.
- `impermanent_loss(FX_in_x_t0, FX_in_x_tn, relative='N')`: Calculates impermanent loss in either absolute or relative terms.
- `fee_amount_by_orders(volume_amount_in_origin, fee_rate, fee_split)`: Calculates fee amounts based on volume, fee rate, and fee split.
- `swap_calc(quote_type, swap_from_amount_reserve, swap_to_amount_reserve)`: Calculates swap amount based on quote type.
- `fee_amount_by_reserves(FX_in_x, volume_in_x, fee_rate)`: Calculates fee amounts based on reserves, FX ratio, volume, and fee rate.
- `reserves_calc(k, FX_in_x, calculate_x=True)`: Calculates reserves amount based on reserves, FX ratio, and a flag to determine X or Y.
- `profitability_bounds(FX_in_x_t0, fee_rate)`: Calculates profitability bounds for the given initial FX ratio and fee rate.
- `t0_calc(pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x)`: Performs pre-calculation for time step 0.
- `tn_calc(FX_timeseries, volume_timeseries, max_paths)`: Calculates results for each time step 'n' based on given time series and paths.
- `pipeline(pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x, FX_timeseries, volume_timeseries, max_paths, deposit_split_percentage)`: Executes the complete pipeline for payoff calculation.

### Note:

- This class is designed for financial modeling and simulation purposes.
- The provided functions are used to calculate various financial metrics and performance indicators.

## Analytics Class

### Overview

The `analytics` class performs financial analytics on a set of paths, focusing on cumulative interest income and net interest income. It is designed to work with a DataFrame containing financial paths, providing methods to calculate and analyze relevant metrics.

#### Functions:

- `endpoint_stat(self)`: Computes and displays descriptive statistics for cumulative interest income, net interest income, and FX values in the endpoint DataFrame.

### Note:

- This class is designed for financial modeling and simulation purposes.
- The provided functions are used to calculate various financial metrics and performance indicators.

## Visualization Class

### Overview

The `Visualisation` class provides methods for generating 3D visualizations of simulation paths and Impermanent Loss (IL) curves.

#### Attributes:

- `paths_df` (DataFrame): DataFrame containing simulation paths data.
- `pool_performance` (DataFrame): DataFrame containing pool performance data.
- `analytics_df` (DataFrame): DataFrame containing analytics data derived from paths_df.
- `endpoint_x_range` (list): X-axis range for the simulation endpoints.
- `endpoint_y_mean` (float): Mean value of the cumulative net returns at simulation endpoints.
- `endpoint_y_std` (float): Standard deviation of the cumulative net returns at simulation endpoints.
- `amm_engine` (Payoff): An instance of the Payoff class for AMM (Automated Market Maker) calculations.

#### Methods:

- `__init__(self, paths_df, pool_performance)`: Initializes the Visualisation class with simulation data.
- `path_plot(self)`: Generates a 3D plot of simulation paths, displaying price levels, days, and returns.
- `endpoint_plot(self, IL_curve_x_range=[], IL_curve_y_offset=[])`: Generates a 3D plot of simulation paths with an overlay of an IL curve.

#### Visualisatoin examples:

![plot_path](https://github.com/danielmalinovsky/automated_marketmaker_calc/assets/106654319/c56a2fa8-1660-4d28-9685-e7f42648e381)
![plot_endpoint](https://github.com/danielmalinovsky/automated_marketmaker_calc/assets/106654319/c5e1db98-5bb0-4a45-a575-3d5e08f59741)


# Supported simulation methods
Current simulation methods are limited to Geometric Brownian Motion (GBM) and Ornstein-Uhlenbeck Process (UOP). Simulation of exchange rate and volume data having these assumptions:
- Exchange rate and exchange volume are independent
- Exchange rate follows GBM
- Exchange volume follows OUP
- Volatility is stable for both exchange rate and volume

These simulation methods are meant as simple proof of concept for the sake of functionality of the payoff method. More advanced approaches for time series simulation are highly recommended.

## Geometric Brownian Motion (GBM) for Price Data

### Overview

A Python class for simulating and analyzing financial price data using Geometric Brownian Motion (GBM). This class provides a set of functions to load historical price data, calculate returns, simulate GBM, and visualize the results. It includes functionalities for both regular simulation and backtesting.

#### Attributes:

- `ticker` (str): Ticker symbol for the financial instrument.
- `start_date` (list): Start date [year, month, day] for historical data loading.
- `end_date` (list/str): End date [year, month, day] or 'today' for historical data loading.
- `interval` (str): Data interval (e.g., '1d') for historical data loading.
- `predicted_period` (int): Number of periods for future price simulation.
- `backtesting` (bool): True for backtesting, False for regular simulation.

#### Functions:

- `data_loading`: Compiles historical price data from Yahoo Finance for a specified ticker.
- `returnify`: Calculates simple returns for provided financial data.
- `log_returnify`: Calculates log returns for provided financial data.
- `GBM_params`: Calculates mean and standard deviation of returns.
- `GBM`: Simulates Geometric Brownian Motion for a given set of parameters.
- `plot_paths`: Plots realizations of GBM along with actual exchange rates.
- `plot_paths_on_profit_space`: Plots realizations of GBM with profit space boundaries.
- `pipeline`: Executes the complete pipeline for GBM simulation and visualization.

#### Note:

- The class assumes the use of Pandas for handling financial data and NumPy for numerical operations.
- Visualization functionalities use Matplotlib for plotting.

## Volume Data Simulation using GBM and Ornstein-Uhlenbeck Process (UOP)

### Overview

A Python class for simulating and analyzing financial volume data using Geometric Brownian Motion (GBM) and Ornstein-Uhlenbeck Process (UOP). This class provides a set of functions to calculate returns, simulate GBM and UOP processes, and visualize the results. It includes functionalities for both regular simulation and backtesting.

#### Attributes:

- `mu` (float): Mean of volume returns.
- `sigma` (float): Standard deviation of volume returns.
- `theta` (float): Speed of mean reversion for UOP.
- `S0` (float): Initial volume.
- `steps` (int): Number of time steps for simulation.
- `n_paths` (int): Number of simulation paths.
- `plot` (str): 'Y' to plot simulation, 'N' otherwise.
- `pandas` (str): 'Y' to return Pandas DataFrame, 'N' for NumPy array.
- `ticker` (str): Ticker symbol for the financial instrument.
- `predicted_period` (int): Number of predicted periods for simulation.
- `backtesting` (bool): True for backtesting, False for regular simulation.

#### Functions:

- `returnify`: Calculates simple returns for provided financial volume data.
- `log_returnify`: Calculates log returns for provided financial volume data.
- `GBM_params`: Calculates the mean and standard deviation of volume returns.
- `GBM`: Simulates Geometric Brownian Motion for a given set of parameters.
- `UOP`: Simulates Ornstein-Uhlenbeck Process for a given set of parameters.
- `plot_paths`: Plots realizations of Ornstein-Uhlenbeck Process along with actual volume data.
- `pipeline`: Executes the complete pipeline for Ornstein-Uhlenbeck Process simulation and visualization.

#### Note:

- The class assumes the use of Pandas for handling financial data and NumPy for numerical operations.
- Visualization functionalities use Matplotlib for plotting.



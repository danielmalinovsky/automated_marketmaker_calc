import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics
from IPython.display import clear_output
import time
import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class price_sim:

  """
    A Python class for simulating and analyzing financial price data using Geometric Brownian Motion (GBM).

    This class provides a set of functions to load historical price data, calculate returns, simulate GBM,
    and visualize the results. It includes functionalities for both regular simulation and backtesting.

    Attributes:
        - ticker (str): Ticker symbol for the financial instrument.
        - start_date (list): Start date [year, month, day] for historical data loading.
        - end_date (list/str): End date [year, month, day] or 'today' for historical data loading.
        - interval (str): Data interval (e.g., '1d') for historical data loading.
        - predicted_period (int): Number of periods for future price simulation.
        - backtesting (bool): True for backtesting, False for regular simulation.

    Functions:
        - data_loading: Compiles historical price data from Yahoo Finance for a specified ticker.
        - returnify: Calculates simple returns for provided financial data.
        - log_returnify: Calculates log returns for provided financial data.
        - GBM_params: Calculates mean and standard deviation of returns.
        - GBM: Simulates Geometric Brownian Motion for a given set of parameters.
        - plot_paths: Plots realizations of GBM along with actual exchange rates.
        - plot_paths_on_profit_space: Plots realizations of GBM with profit space boundaries.
        - pipeline: Executes the complete pipeline for GBM simulation and visualization.

    Note:
        - The class assumes the use of Pandas for handling financial data and NumPy for numerical operations.
        - Visualization functionalities use Matplotlib for plotting.
  """

  def data_loading(self, ticker, start_date, end_date, interval):

      """
      - Calculates log returns for the provided financial data.

    - Parameters:
          - FX_data (Pandas DataFrame): Financial data.
          - date_col (str, optional): Column containing dates.
        
      - Returns: 
        - Pandas DataFrame with dates and corresponding log returns.
      """

      self.ticker = ticker
      self.start_date = start_date
      self.end_date = end_date
      self.interval = interval

      yh_start_date = int(time.mktime(datetime.datetime(self.start_date[0], self.start_date[1], self.start_date[2], 23, 59).timetuple()))

      if self.end_date == 'today':
        yh_end_date = int(time.mktime(datetime.datetime.now().timetuple()))
      else:
        yh_end_date = int(time.mktime(datetime.datetime(self.end_date[0], self.end_date[1], self.end_date[2], 23, 59).timetuple()))

      query_string1 = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={yh_start_date}&period2={yh_end_date}&interval={self.interval}&events=history&includeAdjustedClose=true'

      FX_data = pd.read_csv(query_string1)
      FX_data['Adj Close'] = FX_data['Adj Close']

      return FX_data

  def returnify(self, FX_data, date_col = None):

      """
      - Calculates simple returns for the provided financial data.
  
     - Parameters:
          - FX_data (Pandas DataFrame): Financial data.
          - date_col (str, optional): Column containing dates.
        
      - Returns: 
          - Pandas DataFrame with dates and corresponding returns.
      """

      self.FX_data = FX_data
      self.date_col = date_col

      if self.date_col != None:
        prices = self.FX_data.loc[:, self.FX_data.columns != self.date_col]
        dates = self.FX_data[date_col]
        returns = prices / prices.shift(1) - 1
        returnified = pd.concat([dates, returns], axis = 1,)

      else:
        prices = self.FX_data
        returns = prices / prices.shift(1) - 1
        returnified = returns

      return returnified

  def log_returnify(self, FX_data, date_col = None):

      """
    - Calculates log returns for the provided financial data.

    - Parameters:
        - FX_data (Pandas DataFrame): Financial data.
        - date_col (str, optional): Column containing dates.
        
    - Returns: 
        - Pandas DataFrame with dates and corresponding log returns.
      """

      self.FX_data = FX_data
      self.date_col = date_col

      if self.date_col != None:
        prices = self.FX_data.loc[:, self.FX_data.columns != self.date_col]
        dates = self.FX_data[self.date_col]
        returns = np.log(prices) - np.log(prices.shift(1))
        returnified = pd.concat([dates, returns], axis = 1,)

      else:
        prices = self.FX_data
        returns = np.log(prices) - np.log(prices.shift(1))
        returnified = returns

      return returnified

  def GBM_params(self, returns):

      """
    - Calculates the mean (mu) and standard deviation (sigma) of returns.

    - Parameters:
        - returns (Pandas Series): Financial returns.
        
    - Returns: 
        - Tuple (mu, sigma) representing mean and standard deviation.
      """

      self.returns = returns

      mu = self.returns.describe().at['mean']
      sigma = self.returns.describe().at['std']

      return mu, sigma

  def GBM(self, mu, sigma, S0, steps, n_paths, plot = 'N', pandas = 'Y'):

      """
    - Simulates Geometric Brownian Motion (GBM) for a given set of parameters.

    - Parameters:
        - mu (float): Mean of returns.
        - sigma (float): Standard deviation of returns.
        - S0 (float): Initial stock price.
        - steps (int): Number of time steps.
        - n_paths (int): Number of simulation paths.
        - plot (str): 'Y' to plot simulation, 'N' otherwise.
        - pandas (str): 'Y' to return Pandas DataFrame, 'N' for NumPy array.
        
    - Returns: 
        - Pandas DataFrame or NumPy array representing GBM simulation.
      """

      self.mu = mu
      self.sigma = sigma
      self.S0 = S0
      self.steps = steps
      self.n_paths = n_paths
      self.plot = plot
      self.pandas = pandas

      T = 1

      # calc each time step
      dt = T/self.steps

      # simulation using numpy arrays
      St = np.exp(
          (self.mu - self.sigma ** 2 / 2) * dt
          + self.sigma * np.random.normal(0, np.sqrt(dt), size=(self.n_paths,self.steps)).T
      )

      # include array of 1's
      St = np.vstack([np.ones(self.n_paths), St])

      # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
      St = self.S0 * St.cumprod(axis=0)

      if self.plot == 'Y':
        # Define time interval correctly
        time = np.linspace(0,T,self.steps+1)

        # Require numpy array that is the same shape as St
        tt = np.full(shape=(self.n_paths,self.steps+1), fill_value=time).T

        plt.plot(tt, St)
        plt.xlabel("Years $(t)$")
        plt.ylabel("Stock Price $(S_t)$")
        plt.title(
            "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(self.S0, self.mu, self.sigma)
        )
        plt.show()

      if self.pandas == 'Y':
        output_St = pd.DataFrame(St)
      else:
        output_St = St

      return output_St

  def plot_paths(self, sim_df, FX_df, predicted_period, backtesting = False):

    """
    - Plots realizations of Geometric Brownian Motion along with actual exchange rates.

    - Parameters:
        - sim_df (Pandas DataFrame): Simulated GBM data.
        - FX_df (Pandas Series): Actual exchange rate data.
        - predicted_period (int): Number of predicted periods.
        - backtesting (bool): True for backtesting, False for regular simulation.
    """

    self.ticker = 'USD-Y'

    self.sim_df = sim_df
    self.FX_df = FX_df
    self.predicted_period = predicted_period
    self.backtesting = backtesting

    time_space = np.linspace(0,1,self.predicted_period+1)
    tt = np.full(shape=(10000, self.predicted_period + 1), fill_value=time_space).T
    tt_fx = np.full(shape=(1, self.predicted_period + 1), fill_value=time_space).T

    temp = self.FX_df[self.FX_df.index >= len(self.FX_df)-self.predicted_period-1].reset_index(drop = True)

    if self.backtesting == True:
      plot_df = self.sim_df
    else:
      df_nan = pd.DataFrame(np.nan, index=range(self.predicted_period), columns=range(len(self.sim_df.columns)))
      plot_df = pd.concat([df_nan, self.sim_df]).reset_index(drop = True)
      temp = pd.concat([temp, df_nan]).reset_index(drop = True)
      time_space = np.linspace(0,1,2*self.predicted_period+1)
      tt = np.full(shape=(10000, 2*self.predicted_period + 1), fill_value=time_space).T
      tt_fx = np.full(shape=(1, 2*self.predicted_period + 1), fill_value=time_space).T

    fig = plt.figure(figsize=(8,5))
    plt.plot(tt, plot_df)
    #plt.plot(tt_fx, temp)#, linewidth = 1, c = 'k')
    plt.xlabel("Years $(t)$")
    plt.ylabel("Exchange Rate $(S_t)$")
    plt.title(
            "Realizations of Geometric Brownian Motion of {3}\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(round(self.S0,6), round(self.mu,6), round(self.sigma,6), self.ticker)
        )
    plt.grid()
    plt.show()

  def plot_paths_on_profit_space(self, sim_df, FX_df, predicted_period, avg_returns, backtesting = False):

    """
    - Plots realizations of Geometric Brownian Motion with profit space boundaries.

    - Parameters:
        - sim_df (Pandas DataFrame): Simulated GBM data.
        - FX_df (Pandas Series): Actual exchange rate data.
        - predicted_period (int): Number of predicted periods.
        - avg_returns (float): Average returns for profit space boundaries.
        - backtesting (bool): True for backtesting, False for regular simulation.
    """

    t_start = 0
    t_end = 1
    t_step = 1/predicted_period

    avg_fee_rate = avg_returns

    profit_space = pd.DataFrame(columns=['t','bounds_x1', 'bounds_x2'])
    profit_space['t'] = pd.Series(np.arange(t_start-t_step,t_end,t_step))

    bounds_x1 = pd.Series(profitability_bounds(FX_df.iloc[-predicted_period-1], 0))
    bounds_x2 = pd.Series(profitability_bounds(FX_df.iloc[-predicted_period-1], 0))


    for i in range(1, len(profit_space)):
        bounds_x1 = pd.concat([bounds_x1, pd.Series(profitability_bounds(FX_df.iloc[-predicted_period-1], (((1+avg_fee_rate)**i)-1))[0])])
        bounds_x2 = pd.concat([bounds_x2, pd.Series(profitability_bounds(FX_df.iloc[-predicted_period-1], (((1+avg_fee_rate)**i)-1))[1])])

    profit_space['bounds_x1'] = bounds_x1.reset_index(drop=True)
    profit_space['bounds_x2'] = bounds_x2.reset_index(drop=True)

    self.ticker = 'USD-Y'

    self.sim_df = sim_df
    self.FX_df = FX_df
    self.predicted_period = predicted_period
    self.backtesting = backtesting

    time_space = np.linspace(0,1,self.predicted_period+1)
    tt = np.full(shape=(10000, self.predicted_period + 1), fill_value=time_space).T
    tt_fx = np.full(shape=(1, self.predicted_period + 1), fill_value=time_space).T

    temp = self.FX_df[self.FX_df.index >= len(self.FX_df)-self.predicted_period-1].reset_index(drop = True)

    if self.backtesting == True:
      plot_df = self.sim_df
    else:
      df_nan = pd.DataFrame(np.nan, index=range(self.predicted_period), columns=range(len(self.sim_df.columns)))
      plot_df = pd.concat([df_nan, self.sim_df]).reset_index(drop = True)
      temp = pd.concat([temp, df_nan]).reset_index(drop = True)
      time_space = np.linspace(0,1,2*self.predicted_period+1)
      tt = np.full(shape=(10000, 2*self.predicted_period + 1), fill_value=time_space).T
      tt_fx = np.full(shape=(1, 2*self.predicted_period + 1), fill_value=time_space).T

    fig = plt.figure(figsize=(8,5))
    plt.plot(tt, plot_df)
    #plt.plot(tt_fx, temp)#, linewidth = 1, c = 'k')
    x = profit_space['t']
    y1 = profit_space['bounds_x1']
    y2 = profit_space['bounds_x2']

    plt.fill_between(x, y1, y2, alpha=0.4)
    plt.xlabel("Years $(t)$")
    plt.ylabel("Exchange Rate $(S_t)$")
    plt.title(
            "Realizations of Geometric Brownian Motion of {3}\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(round(self.S0,6), round(self.mu,6), round(self.sigma,6), self.ticker)
        )
    plt.grid()
    plt.show()

  def pipeline(self, predicted_period, FX_data = pd.DataFrame(), ticker = '', start_date = [], end_date = [], interval = '', backtesting = True, plot_sim = False,):

    """
    - Executes the complete pipeline for Geometric Brownian Motion simulation and visualization.

    - Parameters:
        - predicted_period (int): Number of predicted periods.
        - FX_data (Pandas DataFrame, optional): Historical price data.
        - ticker (str): Ticker symbol.
        - start_date (list): Start date [year, month, day].
        - end_date (list/str): End date [year, month, day] or 'today'.
        - interval (str): Data interval (e.g., '1d').
        - backtesting (bool): True for backtesting, False for regular simulation.
        - plot_sim (bool): True to plot the simulation, False otherwise.
    """

    self.ticker = ticker
    self.start_date = start_date
    self.end_date = end_date
    self.interval = interval
    self.predicted_period = predicted_period
    self.backtesting = backtesting

    if FX_data.empty:
        self.FX_data = self.data_loading(ticker = self.ticker,
                            start_date=self.start_date,
                            end_date=self.end_date,
                            interval=self.interval)
    else:
        self.FX_data = FX_data

    self.returns = self.log_returnify(FX_data = self.FX_data,
                            date_col = 'Date')

    self.mu, self.sigma = self.GBM_params(self.returns['Adj Close'])

    self.sim = self.GBM(
        mu = self.mu*(self.predicted_period**(1/2)),
        sigma = self.sigma*(self.predicted_period**(1/2)),
        S0 = self.FX_data.at[len(self.FX_data)-self.predicted_period-1, 'Adj Close'] if backtesting == True else self.FX_data.at[len(self.FX_data)-1, 'Adj Close'],
        #FX_data.at[len(FX_data)-predicted_period-1, 'Close'],
        steps = self.predicted_period,
        n_paths = 10000)
    self.sim_df = self.sim
    
    if plot_sim:
        self.plot_paths(
            sim_df = self.sim,
            FX_df = self.FX_data['Adj Close'],
            predicted_period = self.predicted_period,
            backtesting = self.backtesting)

class volume_sim:

  """
    A Python class for simulating and analyzing financial volume data using Geometric Brownian Motion (GBM)
    and Ornstein-Uhlenbeck Process (UOP).

    This class provides a set of functions to calculate returns, simulate GBM and UOP processes, and visualize
    the results. It includes functionalities for both regular simulation and backtesting.

    Attributes:
        - mu (float): Mean of volume returns.
        - sigma (float): Standard deviation of volume returns.
        - theta (float): Speed of mean reversion for UOP.
        - S0 (float): Initial volume.
        - steps (int): Number of time steps for simulation.
        - n_paths (int): Number of simulation paths.
        - plot (str): 'Y' to plot simulation, 'N' otherwise.
        - pandas (str): 'Y' to return Pandas DataFrame, 'N' for NumPy array.
        - ticker (str): Ticker symbol for the financial instrument.
        - predicted_period (int): Number of predicted periods for simulation.
        - backtesting (bool): True for backtesting, False for regular simulation.

    Functions:
        - returnify: Calculates simple returns for provided financial volume data.
        - log_returnify: Calculates log returns for provided financial volume data.
        - GBM_params: Calculates the mean and standard deviation of volume returns.
        - GBM: Simulates Geometric Brownian Motion for a given set of parameters.
        - UOP: Simulates Ornstein-Uhlenbeck Process for a given set of parameters.
        - plot_paths: Plots realizations of Ornstein-Uhlenbeck Process along with actual volume data.
        - pipeline: Executes the complete pipeline for Ornstein-Uhlenbeck Process simulation and visualization.

    Note:
        - The class assumes the use of Pandas for handling financial data and NumPy for numerical operations.
        - Visualization functionalities use Matplotlib for plotting.
  """

  def returnify(self, FX_data, date_col = None):

      """
    - Calculates simple returns for the provided financial volume data.

    - Parameters:
        - FX_data (Pandas DataFrame): Financial volume data.
        - date_col (str, optional): Column containing dates.
        
    - Returns: 
        - Pandas DataFrame with dates and corresponding returns.
      """

      self.FX_data = FX_data
      self.date_col = date_col

      if self.date_col != None:
        prices = self.FX_data.loc[:, self.FX_data.columns != self.date_col]
        dates = self.FX_data[date_col]
        returns = prices / prices.shift(1) - 1
        returnified = pd.concat([dates, returns], axis = 1,)

      else:
        prices = self.FX_data
        returns = prices / prices.shift(1) - 1
        returnified = returns

      return returnified

  def log_returnify(self, FX_data, date_col = None):

      """
    - Calculates log returns for the provided financial volume data.

    - Parameters:
        - FX_data (Pandas DataFrame): Financial volume data.
        - date_col (str, optional): Column containing dates.
        
    - Returns: 
        - Pandas DataFrame with dates and corresponding log returns.
      """

      self.FX_data = FX_data
      self.date_col = date_col

      if self.date_col != None:
        prices = self.FX_data.loc[:, self.FX_data.columns != self.date_col]
        dates = self.FX_data[self.date_col]
        returns = np.log(prices) - np.log(prices.shift(1))
        returnified = pd.concat([dates, returns], axis = 1,)

      else:
        prices = self.FX_data
        returns = np.log(prices) - np.log(prices.shift(1))
        returnified = returns

      return returnified

  def GBM_params(self, returns):

      """
    - Calculates the mean (mu) and standard deviation (sigma) of volume returns.

    - Parameters:
        - returns (Pandas Series): Financial volume returns.
        
    - Returns: 
        - Tuple (mu, sigma) representing mean and standard deviation.
      """

      self.returns = returns

      mu = self.returns.describe().at['mean']
      sigma = self.returns.describe().at['std']

      return mu, sigma

  def GBM(self, mu, sigma, S0, steps, n_paths, plot = 'N', pandas = 'Y'):

      """
    - Simulates Geometric Brownian Motion (GBM) for a given set of parameters.

    - Parameters:
        - mu (float): Mean of volume returns.
        - sigma (float): Standard deviation of volume returns.
        - S0 (float): Initial volume.
        - steps (int): Number of time steps.
        - n_paths (int): Number of simulation paths.
        - plot (str): 'Y' to plot simulation, 'N' otherwise.
        - pandas (str): 'Y' to return Pandas DataFrame, 'N' for NumPy array.
        
    - Returns: 
        - Pandas DataFrame or NumPy array representing GBM simulation.
      """

      self.mu = mu
      self.sigma = sigma
      self.S0 = S0
      self.steps = steps
      self.n_paths = n_paths
      self.plot = plot
      self.pandas = pandas

      T = 1

      # calc each time step
      dt = T/self.steps

      # simulation using numpy arrays
      St = np.exp(
          (self.mu - self.sigma ** 2 / 2) * dt
          + self.sigma * np.random.normal(0, np.sqrt(dt), size=(self.n_paths,self.steps)).T
      )

      # include array of 1's
      St = np.vstack([np.ones(self.n_paths), St])

      # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
      St = self.S0 * St.cumprod(axis=0)

      if self.plot == 'Y':
        # Define time interval correctly
        time = np.linspace(0,T,self.steps+1)

        # Require numpy array that is the same shape as St
        tt = np.full(shape=(self.n_paths,self.steps+1), fill_value=time).T

        plt.plot(tt, St)
        plt.xlabel("Years $(t)$")
        plt.ylabel("Stock Price $(S_t)$")
        plt.title(
            "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(self.S0, self.mu, self.sigma)
        )
        plt.show()

      if self.pandas == 'Y':
        output_St = pd.DataFrame(St)
      else:
        output_St = St

      return output_St

  def UOP(self, mu, sigma, theta, S0, steps, n_paths, plot = 'N', pandas = 'Y'):
  # Set the parameters of the Ornstein-Uhlenbeck process

    """
    - Simulates Ornstein-Uhlenbeck Process (UOP) for a given set of parameters.

    - Parameters:
        - mu (float): Mean of volume returns.
        - sigma (float): Standard deviation of volume returns.
        - theta (float): Speed of mean reversion.
        - S0 (float): Initial volume.
        - steps (int): Number of time steps.
        - n_paths (int): Number of simulation paths.
        - plot (str): 'Y' to plot simulation, 'N' otherwise.
        - pandas (str): 'Y' to return Pandas DataFrame, 'N' for NumPy array.
        
    - Returns: 
        - Pandas DataFrame or NumPy array representing UOP simulation.
    """

    self.mu = mu
    self.sigma = sigma
    self.theta = theta
    self.S0 = S0
    self.steps = steps
    self.n_paths = n_paths
    self.plot = plot
    self.pandas = pandas

    T = 1

    # calc each time step
    dt = T/steps

    # Create an array to store the paths
    paths = np.zeros((self.n_paths, self.steps+1))
    paths[:,0] = self.S0

    # Generate the paths
    for i in range(self.steps):
        dw = np.random.normal(scale=np.sqrt(dt), size=self.n_paths)
        paths[:,i+1] = paths[:,i] + self.theta*(mu-paths[:,i])*dt + self.sigma*dw

    # Plot the paths
    if self.plot == 'Y':
      time_steps = np.linspace(0, T, self.steps+1)
      for i in range(self.n_paths):
          plt.plot(time_steps, paths[i])
      plt.xlabel('Time')
      plt.ylabel('Value')
      plt.title('Ornstein-Uhlenbeck Process with Multiple Paths')
      plt.show()

    if self.pandas == 'Y':
      output_St = pd.DataFrame(paths).transpose()
    else:
      output_St = paths.transpose()

    return output_St

  def plot_paths(self, sim_df, FX_df, predicted_period, backtesting = False):

    """
    - Plots realizations of Ornstein-Uhlenbeck Process along with actual volume data.

    - Parameters:
        - sim_df (Pandas DataFrame): Simulated UOP data.
        - FX_df (Pandas Series): Actual volume data.
        - predicted_period (int): Number of predicted periods.
        - backtesting (bool): True for backtesting, False for regular simulation.
    """

    self.ticker = 'USD-Y'

    self.sim_df = sim_df
    self.FX_df = FX_df
    self.predicted_period = predicted_period
    self.backtesting = backtesting

    time_space = np.linspace(0,1,self.predicted_period+1)
    self.tt = np.full(shape=(10000, self.predicted_period + 1), fill_value=time_space).T
    tt_fx = np.full(shape=(1, self.predicted_period + 1), fill_value=time_space).T

    temp = self.FX_df[self.FX_df.index >= len(self.FX_df)-self.predicted_period-1].reset_index(drop = True)

    if self.backtesting == True:
      plot_df = self.sim_df
    else:
      df_nan = pd.DataFrame(np.nan, index=range(self.predicted_period), columns=range(len(self.sim_df.columns)))
      plot_df = pd.concat([df_nan, self.sim_df]).reset_index(drop = True)
      temp = pd.concat([temp, df_nan]).reset_index(drop = True)
      time_space = np.linspace(0,1,2*self.predicted_period+1)
      self.tt = np.full(shape=(10000, 2*self.predicted_period + 1), fill_value=time_space).T
      tt_fx = np.full(shape=(1, 2*self.predicted_period + 1), fill_value=time_space).T

    fig = plt.figure(figsize=(8,5))
    plt.plot(self.tt, plot_df)
    #plt.plot(tt_fx, temp)#, linewidth = 3, c = 'k')
    plt.xlabel("Years $(t)$")
    plt.ylabel("Exchange Volume $(V_t)$")
    plt.title(
            "Realizations of Ornstein-Uhlenbeck Process of {3}\n $dV_t = \\theta (\mu - V_t) dt + \sigma dW_t$ \n $S_0 = {0}, \\theta = {4}, \mu = {1}, \sigma = {2}$".format(round(self.S0,0), round(self.mu,0), round(self.sigma,0), self.ticker, round(self.theta,6))
    )
    plt.grid()
    plt.show()

  def pipeline(self, ticker, volume, predicted_period, backtesting):

    """
    - Executes the complete pipeline for Ornstein-Uhlenbeck Process simulation and visualization.

    - Parameters:
        - ticker (str): Ticker symbol.
        - volume (Pandas Series): Historical volume data.
        - predicted_period (int): Number of predicted periods.
        - backtesting (bool): True for backtesting, False for regular simulation.
    """

    self.ticker = ticker
    #self.start_date = start_date
    #self.end_date = end_date
    #self.interval = interval
    self.predicted_period = predicted_period
    self.backtesting = backtesting
    self.FX_data = volume

    """
    self.FX_data = self.data_loading(ticker = self.ticker,
                              start_date=self.start_date,
                              end_date=self.end_date,
                              interval=self.interval)

    self.returns = self.log_returnify(FX_data = self.FX_data,)
                            #date_col = 'Date')
    """
    self.mu, self.sigma = self.GBM_params(self.FX_data)#.loc[len(self.FX_data) - predicted_period:])

    self.sim = self.UOP(
        mu = self.mu,
        sigma = self.sigma,#*(self.predicted_period**(1/2)),
        theta = 5,
        S0 = self.FX_data.at[len(self.FX_data)-self.predicted_period-1] if backtesting == True else self.FX_data.at[len(self.FX_data)-1],
        #FX_data.at[len(FX_data)-predicted_period-1, 'Close'],
        steps = self.predicted_period,
        n_paths = 10000)

    self.sim_df = self.sim
    """
    self.plot_paths(
        sim_df = self.sim,
        FX_df = self.FX_data,
        predicted_period = self.predicted_period,
        backtesting = self.backtesting)"""

# USD = x = denominator
# Y = y = the other one

class payoff:

    """
    A Python class for financial modeling and simulation related to automated marketmaker payoffs. Please be advised that current version of payoff calculator is using UNISWAP V2 engine.

    Functions:
        - k_product(x, y): Calculates the product of two values x and y.
        - value_in_x(x, y, FX): Calculates the value in X given X and Y amounts and an FX ratio.
        - FX_x_over_y(k_pool, y_amount): Calculates the FX ratio of X over Y for a pool.
        - deposit_split(deposit_value_in_x, percentage_of_x, FX_x_over_y): Splits a deposit value into X and Y amounts based on a given percentage.
        - pool_share(k_depositor, k_pool): Calculates the pool share for a depositor.
        - interest_income(accrued_fee_in_x, value_in_x, relative='N'): Calculates interest income for a depositor.
        - hold_in_x(amount_x_t0, amount_y_t0, FX_tn): Calculates the portfolio value in X.
        - impermanent_loss(FX_in_x_t0, FX_in_x_tn, relative='N'): Calculates impermanent loss in either absolute or relative terms.
        - fee_amount_by_orders(volume_amount_in_origin, fee_rate, fee_split): Calculates fee amounts based on volume, fee rate, and fee split.
        - swap_calc(quote_type, swap_from_amount_reserve, swap_to_amount_reserve): Calculates swap amount based on quote type.
        - fee_amount_by_reserves(FX_in_x, volume_in_x, fee_rate): Calculates fee amounts based on reserves, FX ratio, volume, and fee rate.
        - reserves_calc(k, FX_in_x, calculate_x=True): Calculates reserves amount based on reserves, FX ratio, and a flag to determine X or Y.
        - profitability_bounds(FX_in_x_t0, fee_rate): Calculates profitability bounds for given initial FX ratio and fee rate.
        - t0_calc(pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x): Performs pre-calculation for time step 0.
        - tn_calc(FX_timeseries, volume_timeseries, max_paths): Calculates results for each time step 'n' based on given time series and paths.
        - pipeline(pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x, FX_timeseries, volume_timeseries, max_paths, deposit_split_percentage): Executes the complete pipeline for payoff calculation.

    Note:
        - This class is designed for financial modeling and simulation purposes.
        - The provided functions are used to calculate various financial metrics and performance indicators.
    """

    def __init__(self):

        self.pool_reserves = pd.DataFrame(
            columns=['k', 'amount_x', 'amount_y', 'value_in_x', 'x_fee', 'y_fee']
            )

        self.depositor_reserves = pd.DataFrame(
            columns=['k', 'amount_x', 'amount_y', 'value_in_x', 'x_fee', 'y_fee']
            )

        # Unique Features dataframe

        self.depositor_performance = pd.DataFrame(
            columns=['pool_share', 'interest_income', 'hodl_x', 'impermanent_loss', 'IL_rel', 'II_netto']
        )

        self.pool_performance = pd.DataFrame(
            columns=['FX', 'pool_fee', 'volume']
        )

    def k_product(self, x, y):

        """
        Calculates the product of x and y.

        Parameters:
        - x (float): The first parameter.
        - y (float): The second parameter.

        Returns:
        - float: The product of x and y.
        """

        return(x*y)

    def value_in_x(self, x, y, FX):

        """
        Calculates a value using x, y, and a given exchange rate FX.

        Parameters:
        - x (float): The first parameter.
        - y (float): The second parameter.
        - FX (float): The exchange rate.

        Returns:
        - float: The calculated value.
        """

        return(x + y * FX)

    def FX_x_over_y(self, k_pool, y_amount):

        """
        Calculates the ratio of a pool parameter k_pool to the square of y_amount.

        Parameters:
        - k_pool (float): The pool parameter.
        - y_amount (float): The y amount.

        Returns:
        - float: The calculated ratio.
        """

        return(k_pool / (y_amount**2))

    def deposit_split(self, deposit_value_in_x, percentage_of_x, FX_x_over_y):

        """
        Splits a deposit into x and y components based on certain parameters.

        Parameters:
        - deposit_value_in_x (float): The deposit value in x.
        - percentage_of_x (float): The percentage of x.
        - FX_x_over_y (float): The calculated ratio.

        Returns:
        - tuple: A tuple containing x amount and y amount.
        """

        x_amount = deposit_value_in_x * percentage_of_x
        y_amount = (deposit_value_in_x - x_amount) / FX_x_over_y
        return x_amount, y_amount

    def pool_share(self, k_depositor, k_pool):

        """
        Calculates the pool share based on depositor and pool parameters.

        Parameters:
        - k_depositor (float): The depositor parameter.
        - k_pool (float): The pool parameter.

        Returns:
        - float: The calculated pool share.
        """

        return (math.sqrt(k_depositor)/math.sqrt(k_pool))

    def interest_income(self, accruded_fee_in_x, value_in_x, relative = 'N'):

        """
        Calculates interest income based on accrued fees and portfolio value.

        Parameters:
        - accruded_fee_in_x (float): Accrued fee in x.
        - value_in_x (float): Portfolio value in x.
        - relative (str): 'N' for non-relative, 'Y' for relative.

        Returns:
        - float: The calculated interest income.
        """

        return accruded_fee_in_x/value_in_x

    def hold_in_x(self, amount_x_t0, amount_y_t0, FX_tn):

        """
        Calculates a value based on initial amounts and a given exchange rate.

        Parameters:
        - amount_x_t0 (float): Initial amount in x.
        - amount_y_t0 (float): Initial amount in y.
        - FX_tn (float): The exchange rate.

        Returns:
        - float: The calculated value.
        """

        return amount_x_t0 + amount_y_t0 * FX_tn

    def impermanent_loss(self, FX_in_x_t0, FX_in_x_tn, relative = 'N'):

        """
        Calculates impermanent loss based on different scenarios.

        Parameters:
        - FX_in_x_t0 (float): FX in x at time 0.
        - FX_in_x_tn (float): FX in x at time n.
        - relative (str): 'N' for non-relative, 'Y' for relative.

        Returns:
        - float: The calculated impermanent loss.
        """

        if relative == 'N':
            result = value_in_x - hodl_in_x
        else:
            result = ((2*math.sqrt(FX_in_x_tn/FX_in_x_t0))/(1+(FX_in_x_tn/FX_in_x_t0)))-1
        return result

    def fee_amount_by_orders(self, volume_amount_in_origin, fee_rate, fee_split): 

        """
        Calculates fee amounts based on volume and fee rates.

        Parameters:
        - volume_amount_in_origin (dict): Volume amount in origin.
        - fee_rate (float): The fee rate.
        - fee_split (float): The fee split.

        Returns:
        - tuple: A tuple containing x fee amount and y fee amount.
        """

        x_fee_amount = list(volume_amount_in_origin.values())[0] * fee_rate
        y_fee_amount = list(volume_amount_in_origin.values())[1] * fee_rate
        return x_fee_amount, y_fee_amount

    def swap_calc(self, quote_type, swap_from_amount_reserve , swap_to_amount_reserve): 

        """
        Calculates swap amounts based on different scenarios.

        Parameters:
        - quote_type (str): The type of quote ('ask' or 'bid').
        - swap_from_amount_reserve (list): List containing swap from amount reserve parameters.
        - swap_to_amount_reserve (list): List containing swap to amount reserve parameters.

        Returns:
        - float: The calculated result.
        """

        x = swap_from_amount_reserve[2]
        y = swap_to_amount_reserve[2]

        k = k_product(x, y)

        if quote_type == 'ask':
            y_ask = swap_to_amount_reserve[1]
            x_for_ask = ((k*(y-y_ask))/((y-y_ask)**2))-x
            result = x_for_ask
        elif quote_type == 'bid':
            x_bid = swap_from_amount_reserve[1]
            y_for_bid = y-((k*(x+x_bid))/((x+x_bid)**2))
            result = y_for_bid
        else:
            print('inproper operation')

        return result

    def fee_amount_by_reserves(self, FX_in_x, volume_in_x, fee_rate):

        """
        Calculates fee amounts based on reserves and other parameters.

        Parameters:
        - FX_in_x (float): FX in x.
        - volume_in_x (float): Volume in x.
        - fee_rate (float): The fee rate.

        Returns:
        - tuple: A tuple containing x fee amount and y fee amount.
        """

        y_amount_exchanged = volume_in_x / FX_in_x
        y_fee_amount = y_amount_exchanged / 2 * fee_rate
        x_fee_amount = volume_in_x / 2 * fee_rate

        return x_fee_amount, y_fee_amount

    def reserves_calc(self, k, FX_in_x, calculate_x = True):

        """
        Calculates reserves based on pool parameters and exchange rates.

        Parameters:
        - k (float): The pool parameter.
        - FX_in_x (float): FX in x.
        - calculate_x (bool): A flag to determine if x should be calculated.

        Returns:
        - float: The calculated result.
        """

        if calculate_x == True:
            x_amount = (k * FX_in_x)**(1/2)
            result = x_amount
        elif calculate_x == False:
            y_amount = (k / FX_in_x)**(1/2)
            result = y_amount
        else:
            print('Wrong input in "calculate_x" parameter')

        return result

    def profitability_bounds(self, FX_in_x_t0, fee_rate):

        """
        Calculates profitability bounds based on FX at time 0 and a given fee rate.

        Parameters:
        - FX_in_x_t0 (float): FX in x at time 0.
        - fee_rate (float): The fee rate.

        Returns:
        - tuple: Two values representing profitability bounds (x1, x2).
        """

        y = FX_in_x_t0
        rho = fee_rate

        x1 = (-y+y*(rho**2)-(2*y*rho)+(2*y)*math.sqrt(-rho**2+2*rho))/(-rho**2-1+2*rho) if (-rho**2+2*rho)>=0 else np.inf
        x2 = (-y+y*(rho**2)-(2*y*rho)-(2*y)*math.sqrt(-rho**2+2*rho))/(-rho**2-1+2*rho) if (-rho**2+2*rho)>=0 else -np.inf

        return x1, x2

    def t0_calc(self, pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x):

        """
        Performs pre-calculation for the initial time point for the pool and depositor.

        Parameters:
        - pool_fee (float): The pool fee.
        - amount_x_pool_t0 (float): Initial amount in x for the pool.
        - amount_y_pool_t0 (float): Initial amount in y for the pool.
        - total_investment_x (float): Total investment in x.

        Returns:
        - None
        """

        # t0 pool pre - calculation
        self.pool_performance.at[0, 'pool_fee'] = pool_fee
        self.pool_reserves.at[0, 'k'] = self.k_product(amount_x_pool_t0, amount_y_pool_t0)
        self.pool_performance.at[0, 'FX'] = self.FX_x_over_y(self.pool_reserves.at[0, 'k'], amount_y_pool_t0)

        # t0 depositor pre - calculation
        ### deposit calculation
        self.depositor_reserves.at[0, 'value_in_x'] = total_investment_x
        self.depositor_reserves.at[0, 'amount_x'], self.depositor_reserves.at[0, 'amount_y'] = self.deposit_split(total_investment_x, self.deposit_split_percentage, self.pool_performance.at[0, 'FX'])
        self.depositor_reserves.at[0, 'k'] = self.k_product(self.depositor_reserves.at[0, 'amount_x'], self.depositor_reserves.at[0, 'amount_y'])

        # t0 pool calculation

        self.pool_reserves.at[0, 'amount_x'] = amount_x_pool_t0 + self.depositor_reserves.at[0, 'amount_x']
        self.pool_reserves.at[0, 'amount_y'] = amount_y_pool_t0 + self.depositor_reserves.at[0, 'amount_y']
        self.pool_reserves.at[0, 'k'] = self.k_product(self.pool_reserves.at[0, 'amount_x'], self.pool_reserves.at[0, 'amount_y'])
        self.pool_reserves.at[0, 'value_in_x'] = self.value_in_x(self.pool_reserves.at[0, 'amount_x'], self.pool_reserves.at[0, 'amount_y'], self.pool_performance.at[0, 'FX'])

        # t0 change calculation
        self.pool_performance.at[0, 'FX'] = self.FX_x_over_y(self.pool_reserves.at[0, 'k'], self.pool_reserves.at[0, 'amount_y'])

        #temp_fee_amount = self.fee_amount_by_orders(volume_amount_t0, pool_fee, fee_split_percentage)
        self.pool_reserves.at[0, 'x_fee'] = 0 #temp_fee_amount[0]
        self.pool_reserves.at[0, 'y_fee'] = 0 #temp_fee_amount[1]

        self.depositor_performance.at[0, 'pool_share'] = self.pool_share(self.depositor_reserves.at[0, 'k'], self.pool_reserves.at[0, 'k'])
        self.depositor_reserves.at[0, 'x_fee'] = self.pool_reserves.at[0, 'x_fee'] * self.depositor_performance.at[0, 'pool_share']
        self.depositor_reserves.at[0, 'y_fee'] = self.pool_reserves.at[0, 'y_fee'] * self.depositor_performance.at[0, 'pool_share']

        self.depositor_performance.at[0, 'interest_income'] = self.interest_income(
            self.value_in_x(
                self.depositor_reserves.at[0, 'x_fee'],
                self.depositor_reserves.at[0, 'y_fee'],
                self.pool_performance.at[0, 'FX']
                ),
            self.depositor_reserves.at[0, 'value_in_x']
            )

        self.depositor_performance.at[0, 'IL_rel'] = self.impermanent_loss(
            self.pool_performance.at[0, 'FX'],
            self.pool_performance.at[0, 'FX'],
            relative='Y'
            )

        self.depositor_performance.at[0, 'II_netto'] = self.depositor_performance.at[0, 'interest_income'] - self.depositor_performance.at[0, 'IL_rel']

        self.depositor_performance.at[0, 'hodl_x'] = self.hold_in_x(
            self.depositor_reserves.at[0, 'amount_x'],
            self.depositor_reserves.at[0, 'amount_y'],
            self.pool_performance['FX'][0]
            )
        
    def tn_calc(self, FX_timeseries, volume_timeseries, max_paths):
        
        """
        Calculates payoff over time for multiple paths using price and volume time series data.

        Parameters:
        - FX_timeseries (DataFrame): DataFrame containing FX values over time.
        - volume_timeseries (DataFrame): DataFrame containing volume values over time.
        - max_paths (int): The maximum number of paths.

        Returns:
        - None
        """

        self.paths_df = pd.DataFrame()

        for j in range(1,max_paths):
        #for j in range(1, len(EURUSD_volume.sim_df.iloc[i,:])):

            for i in range(1, len(FX_timeseries.iloc[:,j])):

                # tn pre - calculation
                self.pool_performance.at[i, 'pool_fee'] = self.pool_performance.at[i-1, 'pool_fee']
                self.pool_performance.at[i, 'FX'] = FX_timeseries.iloc[i, j] ####################### Stoch FX input
                self.pool_performance.at[i, 'volume'] = volume_timeseries.iloc[i, j]

                #temp_fee_amount = fee_amount(volume_amount_tn, pool_fee, fee_split_percentage)



                #tn pool calculation
                ### reservers
                self.pool_reserves.at[i, 'amount_x'] = self.reserves_calc(self.pool_reserves.at[i-1, 'k'], self.pool_performance.at[i, 'FX'], calculate_x= True) #- pool_tn.at['pool', 'accruded_fee']['USD'] ####### change here transaction costs
                self.pool_reserves.at[i, 'amount_y'] = self.reserves_calc(self.pool_reserves.at[i-1, 'k'], self.pool_performance.at[i, 'FX'], calculate_x= False) #- pool_tn.at['pool', 'accruded_fee']['y'] ####### change here transaction costs
                
                ### fees in x, fees in y
                temp_fee_amount = self.fee_amount_by_reserves(
                    #(pool_performance.at[i, 'FX'] + pool_performance.at[i-1, 'FX'])/2, fee reserves using average FX in between two days
                    self.pool_performance.at[i, 'FX'],
                    self.pool_performance.at[i, 'volume'],
                    self.pool_performance.at[i, 'pool_fee']
                    )
                self.pool_reserves.at[i, 'x_fee'] = abs(temp_fee_amount[0])
                self.pool_reserves.at[i, 'y_fee'] = abs(temp_fee_amount[1])

                ### updating reserves,k  with collected fees
                self.pool_reserves.at[i, 'amount_x'] = self.pool_reserves.at[i, 'amount_x'] + self.pool_reserves.at[i, 'x_fee']
                self.pool_reserves.at[i, 'amount_y'] = self.pool_reserves.at[i, 'amount_y'] + self.pool_reserves.at[i, 'y_fee']
                self.pool_reserves.at[i, 'k'] = self.k_product(self.pool_reserves.at[i, 'amount_x'], self.pool_reserves.at[i, 'amount_y'])

                ### pool value in x
                self.pool_reserves.at[i, 'value_in_x'] = self.value_in_x(self.pool_reserves.at[i, 'amount_x'], self.pool_reserves.at[i, 'amount_y'], self.pool_performance.at[i, 'FX'])



                #tn depositor calculation
                ### pool share
                self.depositor_performance.at[i, 'pool_share'] = self.depositor_performance.at[i-1, 'pool_share']

                ### reserves, portfolio value
                self.depositor_reserves.loc[i, ['amount_x', 'amount_y', 'value_in_x']] = self.pool_reserves.loc[i, ['amount_x', 'amount_y', 'value_in_x']].values * self.depositor_performance.at[i, 'pool_share']

                ### fees in x, fees in y
                self.depositor_reserves.at[i, 'x_fee'] = self.pool_reserves.at[i, 'x_fee'] * self.depositor_performance.at[i, 'pool_share']
                self.depositor_reserves.at[i, 'y_fee'] = self.pool_reserves.at[i, 'y_fee'] * self.depositor_performance.at[i, 'pool_share']

                ### k product
                self.depositor_reserves.at[i, 'k'] = self.k_product(self.depositor_reserves.at[i, 'amount_x'], self.depositor_reserves.at[i, 'amount_y'])

                ### fees in percentage %
                self.depositor_performance.at[i, 'interest_income'] = self.interest_income(
                self.value_in_x(
                    self.depositor_reserves.loc[0:i,'x_fee'].sum(), 
                    self.depositor_reserves.loc[0:i,'y_fee'].sum(), 
                    self.pool_performance.at[i, 'FX']), 
                self.depositor_reserves.at[0, 'value_in_x'])

                ### impermanent loss in percentage %
                self.depositor_performance.at[i, 'IL_rel'] = self.impermanent_loss(
                    self.pool_performance.at[0, 'FX'], #pool_performance.at[i-1, 'FX'],
                    self.pool_performance.at[i, 'FX'],
                    relative='Y'
                    )
                
                ### net income from fees in percentage %
                self.depositor_performance.at[i, 'II_netto'] = self.depositor_performance.at[i, 'interest_income'] + self.depositor_performance.at[i, 'IL_rel']

                ### benchmark portfolio value
                self.depositor_performance.at[i, 'hodl_x'] = self.hold_in_x(
                    self.depositor_reserves.at[i-1, 'amount_x'],
                    self.depositor_reserves.at[i-1, 'amount_y'],
                    self.pool_performance.at[i,'FX']
                    )

            temp = pd.DataFrame()
            temp = pd.DataFrame(
                {'dfs{0}'.format(j) : [
                    self.pool_performance,
                    self.depositor_performance,
                    self.pool_reserves,
                    self.depositor_reserves
                    ]
                },
                index = [
                    'pool_performance',
                    'depositor_performance',
                    'pool_reserves',
                    'depositor_reserves'])

            self.merge_df = pd.concat([self.pool_performance, self.depositor_performance, self.pool_reserves.add_suffix('_pool'), self.depositor_reserves.add_suffix('_depositor')], axis = 1)
            self.merge_df = self.merge_df.replace(np. nan,0)

            total = max_paths+1
            perc = ((j+1)/total)*100
            if int(perc%1) == 0:
                clear_output()
                print(str(int(perc))+'%')#, end="\n")

            self.time_step_df = pd.DataFrame({'sim':[j], 'dfs':[self.merge_df.copy()]})
            self.paths_df = pd.concat([self.paths_df, self.time_step_df])

            self.paths_df = self.paths_df.reset_index(drop = True)

    def progress_bar(current, total, bar_length=20):

        """ 
        Displays a progress bar to track progress.

        Parameters:
        - current (int): The current progress value.
        - total (int): The total progress value.
        - bar_length (int): The length of the progress bar.

        Returns:
        - None
        """

        fraction = current / total

        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '

        ending = '\n' if current == total else '\r'

        print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    def pipeline(self, pool_fee, amount_x_pool_t0, amount_y_pool_t0, total_investment_x, FX_timeseries, volume_timeseries, max_paths, deposit_split_percentage):

        """
        Executes a pipeline of calculations for pool and depositor over time.

        Parameters:
        - pool_fee (float): The pool fee.
        - amount_x_pool_t0 (float): Initial amount in x for the pool.
        - amount_y_pool_t0 (float): Initial amount in y for the pool.
        - total_investment_x (float): Total investment in x.
        - FX_timeseries (DataFrame): DataFrame containing FX values over time.
        - volume_timeseries (DataFrame): DataFrame containing volume values over time.
        - max_paths (int): The maximum number of paths.
        - deposit_split_percentage (float): The percentage used for deposit split.

        Returns:
        - None
        """

        self.pool_fee = pool_fee
        self.amount_x_pool_t0 = amount_x_pool_t0
        self.amount_y_pool_t0 = amount_y_pool_t0
        self.total_investment_x = total_investment_x
        self.FX_timeseries = FX_timeseries
        self.volume_timeseries = volume_timeseries
        self.max_paths = max_paths
        self.deposit_split_percentage = deposit_split_percentage

        self.t0_calc(
            pool_fee= 0.01,
            amount_x_pool_t0= self.amount_x_pool_t0,
            amount_y_pool_t0= self.amount_y_pool_t0,
            total_investment_x= self.total_investment_x
        )

        self.tn_calc(FX_timeseries = self.FX_timeseries,
            volume_timeseries = self.volume_timeseries, 
            max_paths = self.max_paths
        )

class analytics:

    """
    Description:
        The `analytics` class performs financial analytics on a set of paths, focusing on cumulative interest income and net interest income. It is designed to work with a DataFrame containing financial paths, providing methods to calculate and analyze relevant metrics.

    Functions:
        - endpoint_stat(self): Computes and displays descriptive statistics for cumulative interest income, net interest income, and FX values in the endpoint DataFrame.

      
    Note:
        - This class is designed for financial modeling and simulation purposes.
        - The provided functions are used to calculate various financial metrics and performance indicators.
    """

    def __init__(self, paths_df):
        self.paths_df = paths_df

        self.endpoint_df = pd.DataFrame()

        for j in range(len(self.paths_df)):
            #print(j)
            cum_II = (1+self.paths_df.iloc[j,1]['interest_income']).tail(1) 
            cum_II_netto = (1+self.paths_df.iloc[j,1]['II_netto']).tail(1)

            #print(cum_II)
            cum_II_row = pd.DataFrame([float(cum_II)], columns = ['cum_II'])
            cum_II_netto_row = pd.DataFrame([float(cum_II_netto)], columns = ['cum_II_netto'])

            #print(cum_II_row)
            endpoint_row = pd.concat([self.paths_df.iloc[j,1].tail(1).reset_index(drop=True), cum_II_row, cum_II_netto_row],axis = 1)
            #endpoint_netto_row = pd.concat([paths_df.iloc[j,1].tail(1).reset_index(drop=True), cum_II_netto_row],axis = 1)

            #print(endpoint_row)
            self.endpoint_df = pd.concat([self.endpoint_df, endpoint_row])

            #endpoint_df['cum_II'] = paths_df.iloc[j,1]['interest_income'].cumsum().tail(1)

        self.endpoint_df = self.endpoint_df.reset_index(drop = True)

    def endpoint_stat(self):

        """
        Computes and displays descriptive statistics for cumulative interest income, 
        net interest income, and FX values in the endpoint DataFrame.

        Parameters:
            None

        Returns:
            None
        """

        self.endpoint_df[['cum_II', 'cum_II_netto', 'FX']].describe()

class visualisation:

    """
    The Visualisation class provides methods for generating 3D visualizations of simulation paths and Impermanent Loss (IL) curves.

    Attributes:
    - paths_df (DataFrame): DataFrame containing simulation paths data.
    - pool_performance (DataFrame): DataFrame containing pool performance data.
    - analytics_df (DataFrame): DataFrame containing analytics data derived from paths_df.
    - endpoint_x_range (list): X-axis range for the simulation endpoints.
    - endpoint_y_mean (float): Mean value of the cumulative net returns at simulation endpoints.
    - endpoint_y_std (float): Standard deviation of the cumulative net returns at simulation endpoints.
    - amm_engine (Payoff): An instance of the Payoff class for AMM (Automated Market Maker) calculations.

    Methods:
    - __init__(self, paths_df, pool_performance): Initializes the Visualisation class with simulation data.
    - path_plot(self): Generates a 3D plot of simulation paths, displaying price levels, days, and returns.
    - endpoint_plot(self, IL_curve_x_range=[], IL_curve_y_offset=[]): Generates a 3D plot of simulation paths with an overlay of an IL curve.
    """

    def __init__(self, paths_df, pool_performance):

        """
        Initializes the Visualisation class.

        Parameters:
        - paths_df (DataFrame): DataFrame containing simulation paths data.
        - pool_performance (DataFrame): DataFrame containing pool performance data.
        """

        self.paths_df = paths_df
        self.pool_performance = pool_performance
        self.analytics_df = analytics(self.paths_df)
        self.endpoint_x_range = [math.floor(min(self.analytics_df.endpoint_df['FX'])*100)/100, math.ceil(max(self.analytics_df.endpoint_df['FX'])*100)/100]
        self.endpoint_y_mean = statistics.mean(self.analytics_df.endpoint_df['cum_II_netto'])
        self.endpoint_y_std = statistics.stdev(self.analytics_df.endpoint_df['cum_II_netto'])
        self.amm_engine = payoff()

    def path_plot(self):

        """
        Generates a 3D plot of simulation paths, displaying price levels, days, and returns.
        """

        # Visualisation

        #x, y = np.meshgrid(pool_performance['FX'], list(pool_performance.index))

        total = len(self.paths_df)+1

        self.fig = plt.figure()
        #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.fig.set_size_inches(18.5, 10.5)
        self.ax = plt.axes(projection='3d')
        #ax = fig.gca(projection='3d')

        x_0, y_0 = np.mgrid[500:2000:10j, 0:95:10j]

        y = list(self.pool_performance.index)

        for j in range(len(self.paths_df)):
        #print(j)

            x = self.paths_df.iloc[j,1]['FX']

            z = pd.DataFrame()
            z = 1 + self.paths_df.iloc[j,1]['II_netto']
            #z = z.cumprod()
            z_0 = np.ones_like(x_0)

            #print(x)
            """
            for i in range(len(depositor_performance['II_netto'])):
                z = z.append(depositor_performance['II_netto'][i])
            """

            perc = ((j+1)/total)*100
            if int(perc%1) == 0:
                clear_output()
                print(str(int(perc))+'%')#, end="\n")
            #z_0 = np.zeros((len(impermanent_loss),len(impermanent_loss)))

            self.ax.plot3D(x, y, z)
            #ax.scatter3D(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        self.ax.set_xlabel('Price Level')
        self.ax.set_ylabel('Days')
        self.ax.set_zlabel('Returns')

    def endpoint_plot(self, IL_curve_x_range = [], IL_curve_y_offset = []):

        """
        Generates a scatter plot of simulation path endpoints with an overlay of an Impermanent Loss (IL) curve.

        Parameters:
        - IL_curve_x_range (list, optional): X-axis range for the IL curve. Default is calculated from data.
        - IL_curve_y_offset (list, optional): Y-axis offset for the IL curve. Default is calculated from data.
        """

        # endpoint extraction
        FX_t0 = self.paths_df.iloc[0,1]['FX'][0]

        # IL curve definition
        if not IL_curve_x_range:
            IL_curve_start = self.endpoint_x_range[0] 
            IL_curve_end = self.endpoint_x_range[1] 
            IL_curve_step = (IL_curve_end - IL_curve_start)/100
        else:
            IL_curve_start = IL_curve_x_range[0] 
            IL_curve_end = IL_curve_x_range[1] 
            IL_curve_step = (IL_curve_end - IL_curve_start)/100

        IL_curve = pd.DataFrame(columns=['FX', 'returns'])
        IL_curve['FX'] = pd.Series(np.arange(IL_curve_start,IL_curve_end,IL_curve_step))

        if not IL_curve_y_offset:
            peak_stdev = self.endpoint_y_std 
            peak_mean = self.endpoint_y_mean 
            peak = (peak_mean + 3*peak_stdev) 
        else:
            peak = IL_curve_y_offset[0]

        # IL curve construction
        returns = pd.Series(self.amm_engine.impermanent_loss(FX_t0, IL_curve.at[0,'FX'],relative='Y') + peak)

        for i in range(1, len(IL_curve)):
            returns = pd.concat([returns, pd.Series(self.amm_engine.impermanent_loss(FX_t0, IL_curve.at[i,'FX'],relative='Y') + peak)])

        IL_curve['returns'] = returns.reset_index(drop=True)

        # Visualisation

        #x, y = np.meshgrid(pool_performance['FX'], list(pool_performance.index))

        total = len(self.paths_df)+1

        self.fig_endpoint = plt.figure()
        #fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        self.fig_endpoint.set_size_inches(18.5, 10.5)
        self.ax_endpoint = plt.axes(projection='3d')
        #ax = fig.gca(projection='3d')

        x_0, y_0 = np.mgrid[500:2000:10j, 0:95:10j]

        y = 260#list(pool_performance.index)

        for j in range(len(self.paths_df)):
            print(j)

            x = self.paths_df.iloc[j,1]['FX'][260]

            z = pd.DataFrame()
            z = 1 + self.paths_df.iloc[j,1]['II_netto']
            #z = z.cumprod()
            z = z[260]
            z_0 = np.ones_like(x_0)

            print(x)
            """
            for i in range(len(depositor_performance['II_netto'])):
                z = z.append(depositor_performance['II_netto'][i])
            """

            perc = ((j+1)/total)*100
            if int(perc%1) == 0:
                clear_output()
                print(str(int(perc))+'%')#, end="\n")
            #z_0 = np.zeros((len(impermanent_loss),len(impermanent_loss)))

            self.ax_endpoint.scatter3D(x, y, z)
            #ax.scatter3D(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        self.ax_endpoint.plot(IL_curve['FX'],np.full(len(IL_curve),260), IL_curve['returns'], linewidth=3)

        self.ax_endpoint.set_xlabel('Price Level')
        self.ax_endpoint.set_ylabel('Days')
        self.ax_endpoint.set_zlabel('Returns')

class one_pipeline:

    """
    The one_pipeline class represents a simulation pipeline for evaluating the performance of an Automated Market Maker (AMM) pool.

    Attributes:
    - price (PriceSim): An instance of the PriceSim class for simulating price movements.
    - volume (VolumeSim): An instance of the VolumeSim class for simulating volume movements.
    - amm (Payoff): An instance of the Payoff class for AMM calculations.
    
    Methods:
    - __init__(
        self, 
        pool_fee, 
        amount_x_pool_t0, 
        amount_y_pool_t0, 
        total_investment_x, 
        # ... (Other parameters)
    ): Initializes the OnePipeline class with parameters for simulating and evaluating the AMM pool.
    
    - run_pipeline(self): Runs the simulation pipeline and generates visualizations of simulation paths and Impermanent Loss (IL) curves.
    """

    def __init__(
        self, 
        pool_fee, 
        amount_x_pool_t0, 
        amount_y_pool_t0, 
        total_investment_x, 
        max_paths, 
        deposit_split_percentage, 
        predicted_period, 
        volume,
        captured_volume_perc = 1,
        FX_data = pd.DataFrame(), 
        ticker = '', 
        start_date = [], 
        end_date = [], 
        interval = '', 
        backtesting = True, 
        plot_sim = False
        ):

        self.price = price_sim()
        self.volume = volume_sim()
        self.amm = payoff()

        self.price.pipeline(FX_data = FX_data,
                    ticker = 'EURUSD=X',
                    start_date=[2013,8,12],
                    end_date='today',
                    interval='1d',
                    predicted_period = 260,
                    backtesting = True)
        
        self.volume.pipeline(ticker = 'EURUSD=X',
                    volume = volume,
                    predicted_period = 260,
                    backtesting = True)
        
        self.amm.pipeline(
            pool_fee= pool_fee,
            amount_x_pool_t0= amount_x_pool_t0,
            amount_y_pool_t0= amount_y_pool_t0,
            total_investment_x= total_investment_x,
            FX_timeseries = self.price.sim_df,
            volume_timeseries = self.volume.sim*captured_volume_perc, 
            max_paths = max_paths,
            deposit_split_percentage = deposit_split_percentage
        )

        plot = visualisation(
            paths_df = self.amm.paths_df,
            pool_performance = self.amm.pool_performance
        )

        plot.path_plot()

        plot.endpoint_plot()


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
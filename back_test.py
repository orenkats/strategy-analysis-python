import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta 

# Load the CSV file
data = pd.read_csv(r"C:\Users\User\Desktop\Back_Test_Script\NQ.csv", usecols=['ts_event', 'open', 'high', 'low', 'close', 'volume'])

data.rename(columns={'ts_event': 'datetime'}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])  # This will handle the timezone information automatically


data.set_index('datetime', inplace=True)

# Filter data to include only the last half-year (or recent date range)
data = data[((data.index.year >= 2021) & (data.index.month >= 1))]  

# Resample to 15-Minute Candles for Visualization and Analysis
resampled_data = data.resample('15T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

resampled_data['hlc3'] = (resampled_data['high'] + resampled_data['low'] + resampled_data['close'])/3

# Calculate Pivot Points on Resampled Data
def calculate_pivots(data):
    # Calculate Hourly Pivot Point (ppH)
    hourly_high = data['high'].resample('1H').max()         # Highest price in each hour
    hourly_low = data['low'].resample('1H').min()           # Lowest price in each hour
    hourly_close = data['close'].resample('1H').last()      # Last price in each hour

    
    ppH = (hourly_high + hourly_low + hourly_close) / 3
    data['ppH'] = ppH.reindex(data.index, method='ffill')

    daily_high = data['high'].resample('1D').max()          # Highest price in each day
    daily_low = data['low'].resample('1D').min()            # Lowest price in each day
    daily_close = data['close'].resample('1D').last()       # Last price in each day

    ppD = (daily_high + daily_low + daily_close) / 3
    data['ppD'] = ppD.reindex(data.index, method='ffill')
    
    return data

resampled_data = calculate_pivots(resampled_data)

# Apply Dynamic hlc3_h Calculation on Resampled Data
def calculate_dynamic_hlc3_h(data):
   
    data['hour'] = data.index.floor('H')  # Floor datetime to the nearest hour

    data['high_h'] = data.groupby('hour')['high'].cummax()
    data['low_h'] = data.groupby('hour')['low'].cummin()
    data['hlc3_h'] = (data['high_h'] + data['low_h'] + data['close']) / 3

    # Calculate entry conditions based on hlc3_h
    data['crossHlcLong'] = (data['open'] <= data['hlc3_h']) & (data['close'] > data['hlc3_h'])
    data['crossHlcLongFC'] = (data['low'] <= data['hlc3_h'].shift(1)) & (data['close'] > data['hlc3_h'].shift(1))
    data['crossHlcShort'] = (data['open'] >= data['hlc3_h']) & (data['close'] < data['hlc3_h'])
    data['crossHlcShortFC'] = (data['high'] >= data['hlc3_h'].shift(1)) & (data['close'] < data['hlc3_h'].shift(1))
    data.drop(columns=['hour'], inplace=True)

    print("Resampled Data with Dynamic hlc3_h Calculations:\n", data[['high_h', 'low_h', 'close', 'hlc3_h', 'crossHlcLong', 'crossHlcLongFC', 'crossHlcShort', 'crossHlcShortFC']].head(10))

    return data


resampled_data = calculate_dynamic_hlc3_h(resampled_data)

# Parameters
atr_len = 3
avg_volume_len = 20
sl_mult = 1  
tsl_mult = 1.5 
rr_mult_long = 1.5  
rr_mult_short = 1.5  
pointsMult = 20


# Use `ta` to calculate technical indicators
resampled_data['atr'] = ta.volatility.AverageTrueRange(
    high=resampled_data['high'],
    low=resampled_data['low'],
    close=resampled_data['close'],
    window=atr_len
).average_true_range()

resampled_data['avg_volume'] = ta.trend.SMAIndicator(
    close=resampled_data['volume'],
    window=20
).sma_indicator()

# Liquidation Candle Condition
resampled_data['liquidation_candle'] = (
    (resampled_data['volume'] > resampled_data['avg_volume'] * 2) &
    (resampled_data['high'] - resampled_data['low'] > resampled_data['atr']) &
    (resampled_data.index.hour > 0)
)

# Trading Hours Condition
resampled_data['is_trading_hours'] = (
    ((resampled_data.index.hour >= 5) & (resampled_data.index.hour <= 12)) |
    ((resampled_data.index.hour >= 13) & (resampled_data.index.hour < 20))
)

resampled_data['end_day'] = ((resampled_data.index.hour >= 21) & (resampled_data.index.minute >= 45))
# Define candle conditions based on crossHlcLong and crossHlcShort
resampled_data['candle_cond_long'] = np.where(resampled_data.index.minute == 0 ,resampled_data['crossHlcLongFC'], resampled_data['crossHlcLong'])
resampled_data['candle_cond_short'] = np.where(resampled_data.index.minute == 0 ,resampled_data['crossHlcShortFC'], resampled_data['crossHlcShort'])

# Buy and Sell Signals
resampled_data['buy_signal'] = resampled_data['candle_cond_long'] & resampled_data['is_trading_hours']
resampled_data['sell_signal'] = resampled_data['candle_cond_short'] & resampled_data['is_trading_hours']

# Initialize columns for SL, TP, and other trading logic
resampled_data['position_size'] = 0
resampled_data['stop_loss_long'] = np.nan
resampled_data['stop_loss_short'] = np.nan
resampled_data['pt_long'] = np.nan
resampled_data['pt_short'] = np.nan

# Add a column to track active limit orders
resampled_data['active_limit_long'] = False
resampled_data['active_limit_short'] = False


# Lists to store entry and exit markers for plotting
long_entry_indices = []
long_entry_prices = []
short_entry_indices = []
short_entry_prices = []
long_exit_indices = []
long_exit_prices = []
short_exit_indices = []
short_exit_prices = []
initial_equity = 25000
commission_per_contract = 2  # $2 per contract
total_commissions = 0
equity = initial_equity
trades = []
closed_profits = []  # List to store closed trade profits for Sharpe and Sortino ratios

# Trading logic with entry and exit marker assignment
for i in range(1, len(resampled_data) - 1):
    prev_row = resampled_data.iloc[i - 1]
    row = resampled_data.iloc[i]
    next_row = resampled_data.iloc[i + 1]

    # Initialize or carry forward position size and stop levels
    position_size = prev_row['position_size']
    entry_price = prev_row['entry_price'] if 'entry_price' in prev_row else np.nan  # Track the entry price

    # Calculate SL levels based on current position
    if prev_row['position_size'] <= 0:  # No open long position
        stop_loss_long = round(prev_row['hlc3'] - sl_mult * prev_row['atr'], 2)
    else:  # Open long position; retain previous SL
        stop_loss_long = prev_row['stop_loss_long']

    if prev_row['position_size'] >= 0:  # No open short position
        stop_loss_short = round(prev_row['hlc3'] + sl_mult * prev_row['atr'], 2)
    else:  # Open short position; retain previous SL
        stop_loss_short = prev_row['stop_loss_short']
    
    pt_long = prev_row['pt_long'] if pd.notna(prev_row['pt_long']) else np.nan
    pt_short = prev_row['pt_short'] if pd.notna(prev_row['pt_short']) else np.nan

    # Handle new buy or sell signals before processing current position exits
    if prev_row['buy_signal'] and position_size <= 0:
        limit_order_long = prev_row['hlc3']
        if row['low'] <= limit_order_long:
            # Calculate PnL for a reversal from short to long
            if position_size < 0:  # Closing short to open long
                profit_usd = (entry_price - limit_order_long) * pointsMult 
                equity += profit_usd
                closed_profits.append(profit_usd)
                trades.append({
                    'Date/Time': resampled_data.index[i],
                    'Type': 'Short Exit',
                    'Profit USD': profit_usd
                })
                total_commissions += (commission_per_contract * abs(position_size))
            
            long_entry_indices.append(resampled_data.index[i])
            long_entry_prices.append(limit_order_long)
            position_size = 1
            entry_price = limit_order_long  # Set entry price
            stop_loss_long = round(limit_order_long - sl_mult * prev_row['atr'], 2)
            pt_long = round(limit_order_long + rr_mult_long * (limit_order_long - stop_loss_long), 2)
            total_commissions += (commission_per_contract * abs(position_size))
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Long Entry',
                'Profit USD': None  # Entry, so no profit yet
            })    

    if prev_row['sell_signal'] and position_size >= 0:
        limit_order_short = prev_row['hlc3']
        if row['high'] >= limit_order_short:
            # Calculate PnL for a reversal from long to short
            if position_size > 0:  # Closing long to open short
                profit_usd = (entry_price - limit_order_short) * pointsMult 
                equity += profit_usd
                closed_profits.append(profit_usd)
                trades.append({
                    'Date/Time': resampled_data.index[i],
                    'Type': 'Long Exit',
                    'Profit USD': profit_usd
                })
                total_commissions += (commission_per_contract * abs(position_size))
                
            short_entry_indices.append(resampled_data.index[i])
            short_entry_prices.append(limit_order_short)
            position_size = -1
            entry_price = limit_order_short  # Set entry price
            stop_loss_short = round(limit_order_short + sl_mult * prev_row['atr'], 2)
            pt_short = round(limit_order_short - rr_mult_short * (stop_loss_short - limit_order_short), 2)
            total_commissions += (commission_per_contract * abs(position_size))
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Short Entry',
                'Profit USD': None  # Entry, so no profit yet
            })
    
    # If we are already in a long position, check if it needs to be closed or carried forward
    if position_size > 0:
        if row['low'] <= stop_loss_long:
            # Calculate profit or loss for long position
            profit_usd = (stop_loss_long - entry_price) * pointsMult
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Long Exit (SL)',
                'Profit USD': profit_usd
            })
            total_commissions += (commission_per_contract * abs(position_size))
            long_exit_indices.append(resampled_data.index[i])
            long_exit_prices.append(stop_loss_long)
            position_size = 0
        elif row['high'] >= pt_long:
            # Calculate profit or loss for long position
            profit_usd = (pt_long - entry_price) * pointsMult
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Long Exit (TP)',
                'Profit USD': profit_usd
            })
            total_commissions += (commission_per_contract * abs(position_size))
            long_exit_indices.append(resampled_data.index[i])
            long_exit_prices.append(pt_long)
            position_size = 0
        else:
            # Carry forward the position and update trailing stop if necessary
            trailing_stop = round(max(row['low'] - tsl_mult * row['atr'], stop_loss_long), 2)
            stop_loss_long = trailing_stop

    # If we are already in a short position, check if it needs to be closed or carried forward
    if position_size < 0:
        if row['high'] >= stop_loss_short:
            # Calculate profit or loss for short position
            profit_usd = (entry_price - stop_loss_short) * pointsMult
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Short Exit (SL)',
                'Profit USD': profit_usd
            })
            total_commissions += (commission_per_contract * abs(position_size))
            short_exit_indices.append(resampled_data.index[i])
            short_exit_prices.append(stop_loss_short)
            position_size = 0
        elif row['low'] <= pt_short:
            # Calculate profit or loss for short position
            profit_usd = (entry_price - pt_short) * pointsMult
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'Short Exit (TP)',
                'Profit USD': profit_usd
            })
            total_commissions += (commission_per_contract * abs(position_size))
            short_exit_indices.append(resampled_data.index[i])
            short_exit_prices.append(pt_short)
            position_size = 0
        else:
            # Carry forward the position and update trailing stop if necessary
            trailing_stop = round(min(row['high'] + tsl_mult * row['atr'], stop_loss_short), 2)
            stop_loss_short = trailing_stop

    # Save the updated values back into the DataFrame
    resampled_data.at[resampled_data.index[i], 'position_size'] = position_size
    resampled_data.at[resampled_data.index[i], 'stop_loss_long'] = stop_loss_long
    resampled_data.at[resampled_data.index[i], 'pt_long'] = pt_long
    resampled_data.at[resampled_data.index[i], 'stop_loss_short'] = stop_loss_short
    resampled_data.at[resampled_data.index[i], 'pt_short'] = pt_short
    resampled_data.at[resampled_data.index[i], 'entry_price'] = entry_price  # Save entry price

    # Close all trades at the end of the session
    if row['end_day'] == True:
        if position_size > 0:  # If a long position is open at the end of the session
            profit_usd = (row['close'] - entry_price) * pointsMult  # Calculate PnL for closing at the session's closing price
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'End of Session Long Exit',
                'Profit USD': profit_usd
            })
            long_exit_indices.append(resampled_data.index[i])
            long_exit_prices.append(row['close'])  # Mark the exit at the session's closing price
            total_commissions += (commission_per_contract * abs(position_size))
        
        elif position_size < 0:  # If a short position is open at the end of the session
            profit_usd = (entry_price - row['close']) * pointsMult  # Calculate PnL for closing at the session's closing price
            equity += profit_usd
            closed_profits.append(profit_usd)
            trades.append({
                'Date/Time': resampled_data.index[i],
                'Type': 'End of Session Short Exit',
                'Profit USD': profit_usd
            })
            short_exit_indices.append(resampled_data.index[i])
            short_exit_prices.append(row['close'])  # Mark the exit at the session's closing price
            total_commissions += (commission_per_contract * abs(position_size))
        # Reset position size and active limit orders
        resampled_data.at[resampled_data.index[i], 'position_size'] = 0
        resampled_data.at[resampled_data.index[i], 'active_limit_long'] = False
        resampled_data.at[resampled_data.index[i], 'active_limit_short'] = False

# Display all recorded trades with their details
trade_df = pd.DataFrame(trades)

# Calculate key metrics
net_profit = equity - initial_equity - total_commissions
net_percent_profit = ((equity- initial_equity) / initial_equity) * 100
total_trades = len(trade_df)
profitable_trades = trade_df[trade_df['Profit USD'] > 0]
percent_profitable = len(profitable_trades) / total_trades * 100 if total_trades > 0 else 0
max_drawdown = np.min(np.cumsum(closed_profits) - np.maximum.accumulate(np.cumsum(closed_profits)))
sharpe_ratio = np.mean(closed_profits) / np.std(closed_profits) if np.std(closed_profits) > 0 else np.nan
sortino_ratio = np.mean(closed_profits) / np.std([p for p in closed_profits if p < 0]) if len([p for p in closed_profits if p < 0]) > 0 else np.nan
total_gross_profit = trade_df[trade_df['Profit USD'] > 0]['Profit USD'].sum()
total_gross_loss = -trade_df[trade_df['Profit USD'] < 0]['Profit USD'].sum()  # Make it positive for the denominator

# Calculate Profit Factor
if total_gross_loss > 0:
    profit_factor = total_gross_profit / total_gross_loss
else:
    profit_factor = np.inf if total_gross_profit > 0 else np.nan

trade_df['Profit USD'] = trade_df['Profit USD'].astype('object')
trade_df['Profit USD'].fillna('', inplace=True)

# Print the results
print("\nKey Metrics:")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Net Percent Profit: {net_percent_profit:.2f}%")
print(f"Total Closed Trades: {total_trades}")
print(f"Percent Profitable: {percent_profitable:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Total Commission Paid: ${total_commissions:.2f}")
print("\nTrade Details:")
print(trade_df)

trade_df.to_csv('trade_details.csv', index=False)

# Create a dictionary for the key metrics
key_metrics = {
    'Net Profit': [net_profit],
    'Total Closed Trades': [total_trades],
    'Percent Profitable': [percent_profitable],
    'Profit Factor': [profit_factor],
    'Max Drawdown': [max_drawdown],
    'Sharpe Ratio': [sharpe_ratio],
    'Sortino Ratio': [sortino_ratio],
    'Total Commission Paid': [total_commissions]  # Add this if you calculated total commission
}

# Export the key metrics to a CSV file
key_metrics_df = pd.DataFrame(key_metrics)
key_metrics_df.to_csv('key_metrics.csv', index=False)


# Visualization with Conditional Coloring
def plot_candles(data):
    fig = go.Figure()

    # Plot 15-minute candles
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='15-Min Candles'
    ))

    # Plot Hourly and Daily Pivot Points as step lines
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ppH'].ffill(),
        mode='lines',
        name='Hourly Pivot (ppH)',
        line=dict(color='blue'),
        line_shape='hv'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['ppD'].ffill(),
        mode='lines',
        name='Daily Pivot (ppD)',
        line=dict(color='#22aeb8'),
        line_shape='hv'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['hlc3_h'],
        mode='lines',
        name='hlc3_h',
        line=dict(color='blue',width=1),
        line_shape='hv'  
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stop_loss_long'],
        mode='lines',
        name='Stop Loss Long',
        line=dict(color='green', dash='dot',width = 0.5),  
        line_shape='hv',
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['stop_loss_short'],
        mode='lines',
        name='Stop Loss Short',
        line=dict(color='red', dash='dot',width = 0.5),  
        line_shape='hv',
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=long_entry_indices,
        y=long_entry_prices,
        mode='markers',
        marker=dict(color='blue', size=10, symbol='triangle-up'),
        name='Long Entry'
    ))

    fig.add_trace(go.Scatter(
        x=short_entry_indices,
        y=short_entry_prices,
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Short Entry'
    ))

    # Plot exit markers for long and short positions
    fig.add_trace(go.Scatter(
        x=long_exit_indices,
        y=long_exit_prices,
        mode='markers',
        marker=dict(color='blue', size=5, symbol='x'),  
        name='Long Exit'
    ))

    fig.add_trace(go.Scatter(
        x=short_exit_indices,
        y=short_exit_prices,
        mode='markers',
        marker=dict(color='red', size=5, symbol='x'), 
        name='Short Exit'
    ))
    
    # Customize layout
    fig.update_layout(
        title='15-Minute Candlestick Chart ',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Plot the resampled data
plot_candles(resampled_data)

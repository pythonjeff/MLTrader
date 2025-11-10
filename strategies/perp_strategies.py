import pandas as pd
import pandas_ta_classic as ta

def moving_average_crossover_strategy(df, fast=9, slow=21):
    """
    Moving Average Crossover Strategy:
    Generates signals based on fast and slow EMA crossover.
    LONG when fast EMA crosses above slow EMA,
    SHORT when fast EMA crosses below slow EMA,
    HOLD otherwise.
    """
    df = df.copy()
    df['ema_fast'] = ta.ema(df['close'], length=fast)
    df['ema_slow'] = ta.ema(df['close'], length=slow)
    df['signal'] = 'HOLD'
    
    # Generate signals where crossover occurs
    crossover_long = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    crossover_short = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    
    df.loc[crossover_long, 'signal'] = 'LONG'
    df.loc[crossover_short, 'signal'] = 'SHORT'
    
    return df

def rsi_mean_reversion_strategy(df, period=14, lower=30, upper=70):
    """
    RSI Mean Reversion Strategy:
    LONG when RSI < lower threshold (oversold),
    SHORT when RSI > upper threshold (overbought),
    HOLD otherwise.
    """
    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=period)
    df['signal'] = 'HOLD'
    
    df.loc[df['rsi'] < lower, 'signal'] = 'LONG'
    df.loc[df['rsi'] > upper, 'signal'] = 'SHORT'
    
    return df

def summarize_latest_signal(df):
    """
    Utility function to summarize the latest signal.
    Returns a dictionary with the latest timestamp and signal.
    """
    latest_row = df.iloc[-1]
    return {
        'timestamp': latest_row.name if hasattr(latest_row, 'name') else None,
        'signal': latest_row.get('signal', 'HOLD')
    }

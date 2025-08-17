import talib
import pandas as pd
import pandas_ta as ta
import numpy as np

def calculate_indicators(klines_1h, klines_4h):
    try:
        df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        def calc_indicators(df):
            indicators = {}
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3
            )
            indicators['ma_50'] = talib.SMA(df['close'], timeperiod=50)
            indicators['ma_200'] = talib.SMA(df['close'], timeperiod=200)
            indicators['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            indicators['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            indicators['macd'], indicators['macd_signal'], _ = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            indicators['upper_band'], indicators['middle_band'], indicators['lower_band'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            indicators['kdj'] = ta.kdj(df['high'], df['low'], df['close'], length=9)
            indicators['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            indicators['avl'] = ta.sma(df['volume'], length=14)
            indicators['volume'] = df['volume']
            indicators['obv'] = talib.OBV(df['close'], df['volume'])
            indicators['wr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            return {
                k: float(v.iloc[-1]) if isinstance(v, pd.Series) else float(v[-1]) 
                for k, v in indicators.items() 
                if not isinstance(v, pd.DataFrame)
            }

        return {
            'short_term': calc_indicators(df_1h),
            'long_term': calc_indicators(df_4h)
        }
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {'short_term': {}, 'long_term': {}}

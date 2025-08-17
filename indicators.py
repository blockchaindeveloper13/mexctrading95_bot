import pandas as pd
import pandas_ta as ta
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

def calculate_indicators(klines_1h, klines_4h):
    try:
        df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        def calc_indicators(df):
            indicators = {}
            if TALIB_AVAILABLE:
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
                indicators['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
                indicators['obv'] = talib.OBV(df['close'], df['volume'])
                indicators['wr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            else:
                indicators['rsi'] = ta.rsi(df['close'], length=14)
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
                indicators['stoch_k'] = stoch['STOCHk_14_3_3']
                indicators['stoch_d'] = stoch['STOCHd_14_3_3']
                indicators['ma_50'] = ta.sma(df['close'], length=50)
                indicators['ma_200'] = ta.sma(df['close'], length=200)
                indicators['ema_12'] = ta.ema(df['close'], length=12)
                indicators['ema_26'] = ta.ema(df['close'], length=26)
                indicators['macd'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
                indicators['macd_signal'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACDs_12_26_9']
                bbands = ta.bbands(df['close'], length=20, std=2)
                indicators['upper_band'] = bbands['BBU_20_2.0']
                indicators['middle_band'] = bbands['BBM_20_2.0']
                indicators['lower_band'] = bbands['BBL_20_2.0']
                indicators['sar'] = ta.psar(df['high'], df['low'], af0=0.02, af=0.02, max_af=0.2)['PSARl_0.02_0.2']
                indicators['obv'] = ta.obv(df['close'], df['volume'])
                indicators['wr'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            indicators['kdj'] = ta.kdj(df['high'], df['low'], df['close'], length=9)['K_9_3']
            indicators['avl'] = ta.sma(df['volume'], length=14)
            indicators['volume'] = df['volume']

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

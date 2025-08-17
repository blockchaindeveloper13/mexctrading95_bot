import pandas as pd
import pandas_ta as ta
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_indicators(klines_1h, klines_4h):
    try:
        # Create DataFrame with correct column names
        df_1h = pd.DataFrame(klines_1h, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
        df_4h = pd.DataFrame(klines_4h, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_1h[col] = pd.to_numeric(df_1h[col], errors='coerce')
            df_4h[col] = pd.to_numeric(df_4h[col], errors='coerce')

        # Calculate indicators
        indicators = {}
        
        # 1-hour indicators
        indicators['rsi_1h'] = ta.rsi(df_1h['close'], length=14).iloc[-1]
        indicators['ema_20_1h'] = ta.ema(df_1h['close'], length=20).iloc[-1]
        indicators['ema_50_1h'] = ta.ema(df_1h['close'], length=50).iloc[-1]
        indicators['macd_1h'] = ta.macd(df_1h['close'], fast=12, slow=26, signal=9).iloc[-1]['MACD_12_26_9']
        indicators['macd_signal_1h'] = ta.macd(df_1h['close'], fast=12, slow=26, signal=9).iloc[-1]['MACDs_12_26_9']
        
        # 4-hour indicators
        indicators['rsi_4h'] = ta.rsi(df_4h['close'], length=14).iloc[-1]
        indicators['ema_20_4h'] = ta.ema(df_4h['close'], length=20).iloc[-1]
        indicators['ema_50_4h'] = ta.ema(df_4h['close'], length=50).iloc[-1]
        indicators['macd_4h'] = ta.macd(df_4h['close'], fast=12, slow=26, signal=9).iloc[-1]['MACD_12_26_9']
        indicators['macd_signal_4h'] = ta.macd(df_4h['close'], fast=12, slow=26, signal=9).iloc[-1]['MACDs_12_26_9']
        
        logger.info("Indicators calculated successfully")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

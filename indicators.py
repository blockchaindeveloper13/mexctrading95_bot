import pandas as pd
import pandas_ta as ta
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_indicators(klines_1h, klines_4h):
    try:
        # Klines verisini kontrol et
        if not klines_1h or not klines_4h:
            logger.warning("Empty klines data provided")
            return None

        # 1h verisi için DataFrame oluştur
        df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['close'] = pd.to_numeric(df_1h['close'], errors='coerce')
        df_1h['high'] = pd.to_numeric(df_1h['high'], errors='coerce')
        df_1h['low'] = pd.to_numeric(df_1h['low'], errors='coerce')
        df_1h['volume'] = pd.to_numeric(df_1h['volume'], errors='coerce')
        if df_1h.empty or df_1h['close'].isna().all():
            logger.warning("1h klines DataFrame is empty or contains no valid data")
            return None

        # 4h verisi için DataFrame oluştur
        df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['close'] = pd.to_numeric(df_4h['close'], errors='coerce')
        df_4h['high'] = pd.to_numeric(df_4h['high'], errors='coerce')
        df_4h['low'] = pd.to_numeric(df_4h['low'], errors='coerce')
        df_4h['volume'] = pd.to_numeric(df_4h['volume'], errors='coerce')
        if df_4h.empty or df_4h['close'].isna().all():
            logger.warning("4h klines DataFrame is empty or contains no valid data")
            return None

        # Göstergeleri hesapla
        indicators = {}

        # 1h Göstergeler
        indicators['rsi_1h'] = ta.rsi(df_1h['close'], length=14).iloc[-1] if len(df_1h) >= 14 else None
        indicators['ema_20_1h'] = ta.ema(df_1h['close'], length=20).iloc[-1] if len(df_1h) >= 20 else None
        indicators['ema_50_1h'] = ta.ema(df_1h['close'], length=50).iloc[-1] if len(df_1h) >= 50 else None
        macd_1h = ta.macd(df_1h['close'])
        indicators['macd_1h'] = macd_1h['MACD_12_26_9'].iloc[-1] if not macd_1h.empty and len(macd_1h) >= 1 else None
        indicators['macd_signal_1h'] = macd_1h['MACDs_12_26_9'].iloc[-1] if not macd_1h.empty and len(macd_1h) >= 1 else None
        boll_1h = ta.bbands(df_1h['close'], length=20)
        indicators['bb_upper_1h'] = boll_1h['BBU_20_2.0'].iloc[-1] if not boll_1h.empty and len(boll_1h) >= 1 else None
        indicators['bb_lower_1h'] = boll_1h['BBL_20_2.0'].iloc[-1] if not boll_1h.empty and len(boll_1h) >= 1 else None
        indicators['vwap_1h'] = ta.vwap(df_1h['high'], df_1h['low'], df_1h['close'], df_1h['volume']).iloc[-1] if len(df_1h) >= 1 else None
        stoch_1h = ta.stoch(df_1h['high'], df_1h['low'], df_1h['close'])
        indicators['stoch_k_1h'] = stoch_1h['STOCHk_14_3_3'].iloc[-1] if not stoch_1h.empty and len(stoch_1h) >= 1 else None
        indicators['stoch_d_1h'] = stoch_1h['STOCHd_14_3_3'].iloc[-1] if not stoch_1h.empty and len(stoch_1h) >= 1 else None
        indicators['atr_1h'] = ta.atr(df_1h['high'], df_1h['low'], df_1h['close'], length=14).iloc[-1] if len(df_1h) >= 14 else None

        # 4h Göstergeler
        indicators['rsi_4h'] = ta.rsi(df_4h['close'], length=14).iloc[-1] if len(df_4h) >= 14 else None
        indicators['ema_20_4h'] = ta.ema(df_4h['close'], length=20).iloc[-1] if len(df_4h) >= 20 else None
        indicators['ema_50_4h'] = ta.ema(df_4h['close'], length=50).iloc[-1] if len(df_4h) >= 50 else None
        macd_4h = ta.macd(df_4h['close'])
        indicators['macd_4h'] = macd_4h['MACD_12_26_9'].iloc[-1] if not macd_4h.empty and len(macd_4h) >= 1 else None
        indicators['macd_signal_4h'] = macd_4h['MACDs_12_26_9'].iloc[-1] if not macd_4h.empty and len(macd_4h) >= 1 else None
        boll_4h = ta.bbands(df_4h['close'], length=20)
        indicators['bb_upper_4h'] = boll_4h['BBU_20_2.0'].iloc[-1] if not boll_4h.empty and len(boll_4h) >= 1 else None
        indicators['bb_lower_4h'] = boll_4h['BBL_20_2.0'].iloc[-1] if not boll_4h.empty and len(boll_4h) >= 1 else None
        indicators['vwap_4h'] = ta.vwap(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume']).iloc[-1] if len(df_4h) >= 1 else None
        stoch_4h = ta.stoch(df_4h['high'], df_4h['low'], df_4h['close'])
        indicators['stoch_k_4h'] = stoch_4h['STOCHk_14_3_3'].iloc[-1] if not stoch_4h.empty and len(stoch_4h) >= 1 else None
        indicators['stoch_d_4h'] = stoch_4h['STOCHd_14_3_3'].iloc[-1] if not stoch_4h.empty and len(stoch_4h) >= 1 else None
        indicators['atr_4h'] = ta.atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14).iloc[-1] if len(df_4h) >= 14 else None

        logger.info("Indicators calculated successfully")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

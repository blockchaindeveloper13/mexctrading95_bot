import pandas as pd
import pandas_ta as ta
import logging

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_indicators(klines_1h, klines_4h, order_book=None):
    logger.debug(f"Calculating indicators: klines_1h length={len(klines_1h)}, klines_4h length={len(klines_4h)}, order_book={order_book is not None}")
    try:
        # Klines verisini kontrol et
        if not isinstance(klines_1h, list) or not isinstance(klines_4h, list):
            logger.warning(f"Invalid klines format: klines_1h={type(klines_1h)}, klines_4h={type(klines_4h)}")
            return None
        if not klines_1h or not klines_4h:
            logger.warning("Empty klines data provided")
            return None

        # 1h verisi için DataFrame oluştur
        logger.debug(f"Creating DataFrame for klines_1h: {klines_1h[:5]}")
        df_1h = pd.DataFrame(klines_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        logger.debug(f"1h DataFrame created: {df_1h.head().to_dict()}")
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
        df_1h.set_index('timestamp', inplace=True)
        df_1h.sort_index(inplace=True)
        df_1h['close'] = pd.to_numeric(df_1h['close'], errors='coerce')
        df_1h['high'] = pd.to_numeric(df_1h['high'], errors='coerce')
        df_1h['low'] = pd.to_numeric(df_1h['low'], errors='coerce')
        df_1h['volume'] = pd.to_numeric(df_1h['volume'], errors='coerce')
        if df_1h.empty or df_1h['close'].isna().all():
            logger.warning("1h klines DataFrame is empty or contains no valid data")
            return None

        # 4h verisi için DataFrame oluştur
        logger.debug(f"Creating DataFrame for klines_4h: {klines_4h[:5]}")
        df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        logger.debug(f"4h DataFrame created: {df_4h.head().to_dict()}")
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
        df_4h.set_index('timestamp', inplace=True)
        df_4h.sort_index(inplace=True)
        df_4h['close'] = pd.to_numeric(df_4h['close'], errors='coerce')
        df_4h['high'] = pd.to_numeric(df_4h['high'], errors='coerce')
        df_4h['low'] = pd.to_numeric(df_4h['low'], errors='coerce')
        df_4h['volume'] = pd.to_numeric(df_4h['volume'], errors='coerce')
        if df_4h.empty or df_4h['close'].isna().all():
            logger.warning("4h klines DataFrame is empty or contains no valid data")
            return None

        # Göstergeleri hesapla
        indicators = {}
        logger.debug("Calculating 1h indicators")
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

        logger.debug("Calculating 4h indicators")
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

        logger.debug("Calculating volume changes")
        if len(df_1h) >= 2:
            indicators['volume_change_1h'] = ((df_1h['volume'].iloc[-1] - df_1h['volume'].iloc[-2]) / df_1h['volume'].iloc[-2] * 100) if df_1h['volume'].iloc[-2] != 0 else None
        else:
            indicators['volume_change_1h'] = None

        if len(df_1h) >= 6:
            last_3h_volume = df_1h['volume'].iloc[-3:].sum()
            prev_3h_volume = df_1h['volume'].iloc[-6:-3].sum()
            indicators['volume_change_3h'] = ((last_3h_volume - prev_3h_volume) / prev_3h_volume * 100) if prev_3h_volume != 0 else None
        else:
            indicators['volume_change_3h'] = None

        if len(df_1h) >= 12:
            last_6h_volume = df_1h['volume'].iloc[-6:].sum()
            prev_6h_volume = df_1h['volume'].iloc[-12:-6].sum()
            indicators['volume_change_6h'] = ((last_6h_volume - prev_6h_volume) / prev_6h_volume * 100) if prev_6h_volume != 0 else None
        else:
            indicators['volume_change_6h'] = None

        if len(df_1h) >= 48:
            last_24h_volume = df_1h['volume'].iloc[-24:].sum()
            prev_24h_volume = df_1h['volume'].iloc[-48:-24].sum()
            indicators['volume_change_24h'] = ((last_24h_volume - prev_24h_volume) / prev_24h_volume * 100) if prev_24h_volume != 0 else None
        else:
            indicators['volume_change_24h'] = None

        logger.debug("Calculating bid/ask ratio")
        if order_book:
            bids = sum(float(bid[1]) for bid in order_book.get('bids', [])[:10])
            asks = sum(float(ask[1]) for ask in order_book.get('asks', [])[:10])
            indicators['bid_ask_ratio'] = (bids / asks) if asks != 0 else None
            logger.debug(f"Bid/Ask ratio calculated: {indicators['bid_ask_ratio']}")
        else:
            indicators['bid_ask_ratio'] = None
            logger.debug("No order book provided, bid_ask_ratio set to None")

        logger.info("Indicators calculated successfully")
        logger.debug(f"Returning indicators: {indicators}")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

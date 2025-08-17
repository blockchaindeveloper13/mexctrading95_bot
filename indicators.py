import pandas as pd
import pandas_ta as ta
import logging

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_indicators(klines_60m, klines_4h, order_book=None):
    logger.debug(f"Calculating indicators: klines_60m length={len(klines_60m)}, klines_4h length={len(klines_4h)}, order_book={order_book is not None}")
    try:
        # Klines verisini kontrol et
        if not isinstance(klines_60m, list) or not isinstance(klines_4h, list):
            logger.warning(f"Invalid klines format: klines_60m={type(klines_60m)}, klines_4h={type(klines_4h)}")
            return None
        if not klines_60m or not klines_4h:
            logger.warning("Empty klines data provided")
            return None

        # 60m verisi için DataFrame oluştur
        logger.debug(f"Creating DataFrame for klines_60m: {klines_60m[:5]}")
        df_60m = pd.DataFrame(klines_60m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        logger.debug(f"60m DataFrame created: {df_60m.head().to_dict()}")
        df_60m['timestamp'] = pd.to_datetime(df_60m['timestamp'], unit='ms')
        df_60m.set_index('timestamp', inplace=True)
        df_60m.sort_index(inplace=True)
        df_60m['close'] = pd.to_numeric(df_60m['close'], errors='coerce')
        df_60m['high'] = pd.to_numeric(df_60m['high'], errors='coerce')
        df_60m['low'] = pd.to_numeric(df_60m['low'], errors='coerce')
        df_60m['volume'] = pd.to_numeric(df_60m['volume'], errors='coerce')
        if df_60m.empty or df_60m['close'].isna().all():
            logger.warning("60m klines DataFrame is empty or contains no valid data")
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
        logger.debug("Calculating 60m indicators")
        indicators['rsi_60m'] = ta.rsi(df_60m['close'], length=14).iloc[-1] if len(df_60m) >= 14 else None
        indicators['ema_20_60m'] = ta.ema(df_60m['close'], length=20).iloc[-1] if len(df_60m) >= 20 else None
        indicators['ema_50_60m'] = ta.ema(df_60m['close'], length=50).iloc[-1] if len(df_60m) >= 50 else None
        macd_60m = ta.macd(df_60m['close'])
        indicators['macd_60m'] = macd_60m['MACD_12_26_9'].iloc[-1] if not macd_60m.empty and len(macd_60m) >= 1 else None
        indicators['macd_signal_60m'] = macd_60m['MACDs_12_26_9'].iloc[-1] if not macd_60m.empty and len(macd_60m) >= 1 else None
        boll_60m = ta.bbands(df_60m['close'], length=20)
        indicators['bb_upper_60m'] = boll_60m['BBU_20_2.0'].iloc[-1] if not boll_60m.empty and len(boll_60m) >= 1 else None
        indicators['bb_lower_60m'] = boll_60m['BBL_20_2.0'].iloc[-1] if not boll_60m.empty and len(boll_60m) >= 1 else None
        indicators['vwap_60m'] = ta.vwap(df_60m['high'], df_60m['low'], df_60m['close'], df_60m['volume']).iloc[-1] if len(df_60m) >= 1 else None
        stoch_60m = ta.stoch(df_60m['high'], df_60m['low'], df_60m['close'])
        indicators['stoch_k_60m'] = stoch_60m['STOCHk_14_3_3'].iloc[-1] if not stoch_60m.empty and len(stoch_60m) >= 1 else None
        indicators['stoch_d_60m'] = stoch_60m['STOCHd_14_3_3'].iloc[-1] if not stoch_60m.empty and len(stoch_60m) >= 1 else None
        indicators['atr_60m'] = ta.atr(df_60m['high'], df_60m['low'], df_60m['close'], length=14).iloc[-1] if len(df_60m) >= 14 else None

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
        if len(df_60m) >= 2:
            indicators['volume_change_60m'] = ((df_60m['volume'].iloc[-1] - df_60m['volume'].iloc[-2]) / df_60m['volume'].iloc[-2] * 100) if df_60m['volume'].iloc[-2] != 0 else None
        else:
            indicators['volume_change_60m'] = None

        if len(df_60m) >= 6:
            last_3h_volume = df_60m['volume'].iloc[-3:].sum()
            prev_3h_volume = df_60m['volume'].iloc[-6:-3].sum()
            indicators['volume_change_3h'] = ((last_3h_volume - prev_3h_volume) / prev_3h_volume * 100) if prev_3h_volume != 0 else None
        else:
            indicators['volume_change_3h'] = None

        if len(df_60m) >= 12:
            last_6h_volume = df_60m['volume'].iloc[-6:].sum()
            prev_6h_volume = df_60m['volume'].iloc[-12:-6].sum()
            indicators['volume_change_6h'] = ((last_6h_volume - prev_6h_volume) / prev_6h_volume * 100) if prev_6h_volume != 0 else None
        else:
            indicators['volume_change_6h'] = None

        if len(df_60m) >= 48:
            last_24h_volume = df_60m['volume'].iloc[-24:].sum()
            prev_24h_volume = df_60m['volume'].iloc[-48:-24].sum()
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

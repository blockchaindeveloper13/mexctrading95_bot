import os
import json
import ccxt.async_support as ccxt
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MEXCClient:
    def __init__(self):
        logger.debug("Initializing MEXCClient")
        self.exchange = ccxt.mexc3({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultTimeframe': '60m',
            }
        })
        self.data_file = '/tmp/market_data.json'  # Heroku için /tmp
        logger.debug(f"Data file set to: {self.data_file}")

    async def get_exchange_info(self):
        logger.debug("Fetching exchange info")
        try:
            markets = await self.exchange.load_markets()
            return {'symbols': [{'symbol': k} for k in markets.keys()]}
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {'symbols': []}

    async def get_tickers(self):
        logger.debug("Fetching tickers")
        try:
            return await self.exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    async def get_kline(self, symbol, timeframe, limit=100):
        logger.debug(f"Fetching {timeframe} kline for {symbol}")
        try:
            if timeframe not in ['1m', '5m', '15m', '30m', '60m', '4h', '1d']:  # Desteklenen intervaller
                logger.error(f"Invalid timeframe: {timeframe}")
                raise ValueError(f"Invalid timeframe: {timeframe}")
            klines = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug(f"Fetched {len(klines)} kline entries for {symbol} ({timeframe})")
            return klines
        except Exception as e:
            logger.error(f"Error fetching {timeframe} kline for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol, limit=10):
        logger.debug(f"Fetching order book for {symbol}")
        try:
            return await self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}

    async def fetch_and_save_market_data(self, symbols=None, timeframes=['60m', '4h']):
        logger.info("Starting fetch_and_save_market_data")
        markets = await self.get_exchange_info()
        logger.info(f"{len(markets['symbols'])} market yüklendi")
        usdt_pairs = [m['symbol'] for m in markets['symbols'] if m['symbol'].endswith('USDT')]
        logger.info(f"{len(usdt_pairs)} USDT çifti bulundu (ilk 5): {usdt_pairs[:5]}...")
        
        tickers = await self.get_tickers()
        logger.info(f"{len(tickers)} ticker alındı")
        
        top_symbols = sorted(
            [(symbol, tickers[symbol].get('quoteVolume', 0)) for symbol in usdt_pairs if symbol in tickers],
            key=lambda x: x[1],
            reverse=True
        )
        top_symbols = [symbol for symbol, _ in top_symbols]
        if symbols:
            top_symbols = [s for s in top_symbols if s in symbols]
        else:
            top_symbols = top_symbols[:100]  # Varsayılan olarak top 100
        logger.info(f"En iyi {len(top_symbols)} coin alındı: {top_symbols[:5]}...")
        
        data = []
        for symbol in top_symbols:
            try:
                ticker = tickers.get(symbol, {})
                price = float(ticker.get('lastPrice', 0))
                volume = float(ticker.get('quoteVolume', 0))
                
                klines = {}
                for tf in timeframes:
                    klines[tf] = await self.get_kline(symbol, tf, limit=100)
                
                order_book = await self.get_order_book(symbol, limit=10)
                
                data.append({
                    'coin': symbol,
                    'price': price,
                    'volume': volume,
                    'klines': klines,
                    'order_book': order_book
                })
                logger.info(f"{symbol} için veri alındı: fiyat={price}, hacim={volume}, "
                           f"klines_60m={len(klines.get('60m', []))}, klines_4h={len(klines.get('4h', []))}, "
                           f"order_book_bids={len(order_book.get('bids', []))}")
                
                await asyncio.sleep(0.1)  # API hız sınırı için
            except Exception as e:
                logger.warning(f"{symbol} için veri alınırken hata: {str(e)}")
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"{len(data)} coin {self.data_file}'a kaydedildi")
        
        return data

    async def get_top_coins(self, limit=100, timeframes=['60m', '4h']):
        logger.debug(f"Fetching top {limit} coins with timeframes: {timeframes}")
        try:
            markets = await self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets")
            
            usdt_pairs = [symbol for symbol in markets if symbol.endswith('/USDT')]
            logger.debug(f"Found {len(usdt_pairs)} USDT pairs: {usdt_pairs[:5]}...")
            
            tickers = await self.exchange.fetch_tickers()
            logger.info(f"Fetched {len(tickers)} tickers")
            
            sorted_tickers = sorted(
                [(symbol, tickers[symbol].get('quoteVolume', 0)) for symbol in usdt_pairs if symbol in tickers],
                key=lambda x: x[1],
                reverse=True
            )
            coins = [symbol for symbol, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            
            if not coins:
                logger.warning("No valid USDT pairs found in tickers")
                return []
            
            await self.fetch_and_save_market_data(coins, timeframes)
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol, timeframes=['60m', '4h']):
        logger.debug(f"Fetching market data for {symbol} from JSON with timeframes: {timeframes}")
        try:
            with open(self.data_file, 'r') as f:
                market_data = json.load(f)
            logger.debug(f"Loaded market data from {self.data_file}: {len(market_data)} coins")
            
            for item in market_data:
                if item['coin'] == symbol:
                    data = {
                        'coin': symbol,
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                        'klines': {tf: item['klines'].get(tf, []) for tf in timeframes},
                        'order_book': item.get('order_book', {'bids': [], 'asks': []})
                    }
                    log_msg = f"Fetched market data for {symbol} from JSON: price={data['price']}, volume={data['volume']}"
                    for tf in timeframes:
                        log_msg += f", klines_{tf}={len(data['klines'].get(tf, []))}"
                    log_msg += f", order_book_bids={len(data['order_book'].get('bids', []))}"
                    logger.info(log_msg)
                    return data
            logger.warning(f"No data found for {symbol} in {self.data_file}")
            return None
        except FileNotFoundError:
            logger.error(f"Market data file not found: {self.data_file}")
            return None
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} from JSON: {e}")
            return None

    async def close(self):
        logger.debug("Closing MEXCClient connection")
        await self.exchange.close()
        logger.info("Closed MEXCClient connection")

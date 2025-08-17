import os
import json
import ccxt.async_support as ccxt
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime

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
            'options': {'defaultTimeframe': '60m'}
        })
        self.data_file = '/tmp/market_data.json'

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

    async def get_kline(self, symbol, timeframe, limit=100, retries=3, delay=2):
        logger.debug(f"Fetching {timeframe} kline for {symbol} (retries: {retries})")
        try:
            if timeframe not in ['1m', '5m', '15m', '30m', '60m', '4h', '1d']:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            for attempt in range(retries):
                try:
                    klines = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    if klines:
                        logger.debug(f"Fetched {len(klines)} kline entries for {symbol} ({timeframe})")
                        return klines
                    logger.warning(f"No kline data for {symbol} ({timeframe}) on attempt {attempt + 1}")
                    await asyncio.sleep(delay * (attempt + 1))
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol} ({timeframe}): {e}")
                    if attempt == retries - 1:
                        logger.error(f"Failed to fetch {timeframe} kline for {symbol} after {retries} attempts")
                        return []
                    await asyncio.sleep(delay * (attempt + 1))
            return []
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
        usdt_pairs = [m['symbol'] for m in markets['symbols'] if m['symbol'].endswith('USDT')]
        logger.info(f"{len(usdt_pairs)} USDT pairs found: {usdt_pairs[:5]}...")

        tickers = await self.get_tickers()
        logger.info(f"Fetched {len(tickers)} tickers")

        top_symbols = sorted(
            [(s, tickers[s].get('quoteVolume', 0)) for s in usdt_pairs if s in tickers],
            key=lambda x: x[1], reverse=True
        )
        top_symbols = [s for s, _ in top_symbols]
        if symbols:
            top_symbols = [s for s in top_symbols if s in symbols]
        else:
            top_symbols = top_symbols[:100]
        logger.info(f"Top {len(top_symbols)} symbols: {top_symbols[:5]}...")

        data = []
        for symbol in top_symbols:
            try:
                ticker = tickers.get(symbol, {})
                price = float(ticker.get('lastPrice', 0)) if ticker.get('lastPrice') else 0.0
                volume = float(ticker.get('quoteVolume', 0)) if ticker.get('quoteVolume') else 0.0

                klines = {}
                for tf in timeframes:
                    klines[tf] = await self.get_kline(symbol, tf, limit=100, retries=3, delay=2)
                    if not klines[tf]:
                        logger.warning(f"No {tf} kline data for {symbol}, skipping")
                        break
                if not all(klines.get(tf) for tf in timeframes):
                    continue

                order_book = await self.get_order_book(symbol, limit=10)

                coin_data = {
                    'coin': symbol,
                    'price': price,
                    'volume': volume,
                    'klines': klines,
                    'order_book': order_book
                }
                data.append(coin_data)
                logger.info(f"Data for {symbol}: price={price}, volume={volume}, "
                           f"klines_60m={len(klines.get('60m', []))}, klines_4h={len(klines.get('4h', []))}")

                await asyncio.sleep(0.5)  # Rate limit için
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(data)} coins to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving to {self.data_file}: {e}")

        return data

    async def get_top_coins(self, limit=100, timeframes=['60m', '4h']):
        logger.debug(f"Fetching top {limit} coins")
        try:
            markets = await self.exchange.load_markets()
            usdt_pairs = [s for s in markets if s.endswith('/USDT')]
            tickers = await self.get_tickers()
            sorted_tickers = sorted(
                [(s, tickers[s].get('quoteVolume', 0)) for s in usdt_pairs if s in tickers],
                key=lambda x: x[1], reverse=True
            )
            coins = [s for s, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            if not coins:
                logger.warning("No valid USDT pairs found")
                return []
            await self.fetch_and_save_market_data(coins, timeframes)
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol, timeframes=['60m', '4h']):
        logger.debug(f"Fetching market data for {symbol} from JSON")
        try:
            with open(self.data_file, 'r') as f:
                market_data = json.load(f)
            for item in market_data:
                if item['coin'] == symbol:
                    data = {
                        'coin': symbol,
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                        'klines': {tf: item['klines'].get(tf, []) for tf in timeframes},
                        'order_book': item.get('order_book', {'bids': [], 'asks': []})
                    }
                    logger.info(f"Fetched data for {symbol}: price={data['price']}, "
                               f"klines_60m={len(data['klines']['60m'])}, klines_4h={len(data['klines']['4h'])}")
                    return data
            logger.warning(f"No data for {symbol} in {self.data_file}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_file}")
            return None
        except Exception as e:
            logger.error(f"Error reading market data for {symbol}: {e}")
            return None

    async def close(self):
        logger.debug("Closing MEXCClient")
        await self.exchange.close()
        logger.info("MEXCClient closed")

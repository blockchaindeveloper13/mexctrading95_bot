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
        self.data_file = os.getenv('MARKET_DATA_FILE', 'market_data.json')
        logger.debug(f"Data file set to: {self.data_file}")

   async def fetch_and_save_market_data():
    async with MEXCClient() as client:
        markets = await client.get_exchange_info()
        logger.info(f"{len(markets['symbols'])} market yüklendi")
        usdt_pairs = [m['symbol'] for m in markets['symbols'] if m['symbol'].endswith('USDT')]
        logger.info(f"{len(usdt_pairs)} USDT çifti bulundu (ilk 5): {usdt_pairs[:5]}...")
        
        tickers = await client.get_tickers()
        logger.info(f"{len(tickers)} ticker alındı")
        
        top_100 = sorted(tickers, key=lambda x: float(x['volume']), reverse=True)[:100]
        top_100_symbols = [t['symbol'] for t in top_100]
        logger.info(f"En iyi 100 coin alındı: {top_100_symbols}")
        
        data = {}
        for symbol in top_100_symbols:
            try:
                ticker = next(t for t in tickers if t['symbol'] == symbol)
                price = float(ticker['lastPrice'])
                volume = float(ticker['volume'])
                
                # 60m intervalini kullan
                klines_60m = await client.get_kline(symbol, '60m', limit=100)
                klines_4h = await client.get_kline(symbol, '4h', limit=100)
                order_book = await client.get_order_book(symbol, limit=10)
                
                data[symbol] = {
                    'price': price,
                    'volume': volume,
                    'klines_60m': len(klines_60m),
                    'klines_4h': len(klines_4h),
                    'order_book_bids': len(order_book['bids'])
                }
                logger.info(f"{symbol} için veri alındı: fiyat={price}, hacim={volume}, klines_60m={len(klines_60m)}, klines_4h={len(klines_4h)}, order_book_bids={len(order_book['bids'])}")
                
                # API hız sınırları için kısa bir gecikme
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"{symbol} için veri alınırken hata: {str(e)}")
        
        with open('/tmp/market_data.json', 'w') as f:
            json.dump(data, f)
        logger.info(f"{len(data)} coin market_data.json'a kaydedildi")
        
        return data

    async def get_top_coins(self, limit=10, timeframes=['60m', '4h']):
        logger.debug(f"Fetching top {limit} coins with timeframes: {timeframes}")
        try:
            logger.debug("Loading markets")
            markets = await self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets")
            
            usdt_pairs = [symbol for symbol in markets if symbol.endswith('/USDT')]
            logger.debug(f"Found {len(usdt_pairs)} USDT pairs: {usdt_pairs[:5]}...")
            
            logger.debug("Fetching tickers")
            tickers = await self.exchange.fetch_tickers()
            logger.info(f"Fetched {len(tickers)} tickers")
            
            sorted_tickers = sorted(
                [(symbol, tickers[symbol].get('quoteVolume', 0)) for symbol in usdt_pairs if symbol in tickers],
                key=lambda x: x[1],
                reverse=True
            )
            coins = [symbol for symbol, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins}")
            
            if not coins:
                logger.warning("No valid USDT pairs found in tickers")
                return []
            
            logger.debug(f"Calling fetch_and_save_market_data for {len(coins)} coins")
            await self.fetch_and_save_market_data(coins, timeframes)
            logger.debug(f"Returning {len(coins)} coins")
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
                    logger.debug(f"Returning data for {symbol}: {data}")
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

import os
import ccxt.async_support as ccxt
import logging
from dotenv import load_dotenv

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MEXCClient:
    def __init__(self):
        self.exchange = ccxt.mexc3({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True
        })

    async def get_top_coins(self, limit):
        try:
            # Market verilerini çek
            markets = await self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets")
            
            # Tüm market sembollerini ve örnek market yapısını logla
            all_symbols = list(markets.keys())
            logger.debug(f"All market symbols (first 10): {all_symbols[:10]}")
            if all_symbols:
                logger.debug(f"Sample market data for {all_symbols[0]}: {markets[all_symbols[0]]}")
            
            # USDT çiftlerini filtrele
            usdt_pairs = [symbol for symbol in markets if symbol.endswith('/USDT')]
            logger.info(f"Found {len(usdt_pairs)} USDT pairs (first 5): {usdt_pairs[:5]}...")
            
            # Ticker verilerini çek
            tickers = await self.exchange.fetch_tickers()
            logger.info(f"Fetched {len(tickers)} tickers")
            
            # Hacme göre sırala
            sorted_tickers = sorted(
                [(symbol, tickers[symbol].get('quoteVolume', 0)) for symbol in usdt_pairs if symbol in tickers],
                key=lambda x: x[1],
                reverse=True
            )
            coins = [symbol for symbol, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            if not coins:
                logger.warning("No valid USDT pairs found in tickers")
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            klines_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            klines_4h = await self.exchange.fetch_ohlcv(symbol, '4h', limit=100)
            ticker = await self.exchange.fetch_ticker(symbol)
            order_book = await self.exchange.fetch_order_book(symbol, limit=10)
            data = {
                'coin': symbol,
                'price': ticker.get('last', 0),
                'volume': ticker.get('quoteVolume', 0),
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'order_book': order_book
            }
            logger.info(f"Fetched market data for {symbol}: price={data['price']}, volume={data['volume']}, klines_1h={len(klines_1h)}, klines_4h={len(klines_4h)}, order_book_bids={len(order_book.get('bids', []))}")
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def close(self):
        await self.exchange.close()

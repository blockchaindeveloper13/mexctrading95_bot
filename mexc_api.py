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
        self.exchange = ccxt.mexc({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True
        })

    async def get_top_coins(self, limit):
        try:
            markets = await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            # USDT çiftlerini seç ve hacme göre sırala
            usdt_pairs = [
                symbol for symbol, market in markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
            ]
            sorted_tickers = sorted(
                [(symbol, tickers[symbol]['quoteVolume']) for symbol in usdt_pairs if symbol in tickers],
                key=lambda x: x[1],
                reverse=True
            )
            return [symbol for symbol, _ in sorted_tickers[:limit]]
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            # Klines verisi (1h ve 4h)
            klines_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            klines_4h = await self.exchange.fetch_ohlcv(symbol, '4h', limit=100)
            ticker = await self.exchange.fetch_ticker(symbol)
            # Order book verisi
            order_book = await self.exchange.fetch_order_book(symbol, limit=10)
            return {
                'coin': symbol,
                'price': ticker['last'],
                'volume': ticker['quoteVolume'],
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'order_book': order_book
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def close(self):
        await self.exchange.close()

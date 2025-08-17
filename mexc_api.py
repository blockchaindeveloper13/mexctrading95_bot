import ccxt.async_support as ccxt
import os
from dotenv import load_dotenv

load_dotenv()

class MEXCClient:
    def __init__(self):
        self.client = ccxt.mexc({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True
        })

    async def get_top_coins(self, limit):
        try:
            markets = await self.client.fetch_tickers()
            sorted_markets = sorted(
                markets.items(),
                key=lambda x: float(x[1]['quoteVolume']) if x[1].get('quoteVolume') else 0,
                reverse=True
            )
            top_coins = [symbol for symbol, data in sorted_markets if symbol.endswith('/USDT')][:limit]
            return top_coins
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            klines_1h = await self.client.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            klines_4h = await self.client.fetch_ohlcv(symbol, timeframe='4h', limit=100)
            ticker = await self.client.fetch_ticker(symbol)
            return {
                'coin': symbol,
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'price': ticker['last'],
                'volume': ticker['quoteVolume']
            }
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None

    async def close(self):
        await self.client.close()

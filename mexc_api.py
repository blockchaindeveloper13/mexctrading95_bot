import aiohttp
import os
import hmac
import hashlib
import time
from dotenv import load_dotenv

load_dotenv()

class MEXCClient:
    def __init__(self):
        self.api_key = os.getenv('MEXC_API_KEY')
        self.api_secret = os.getenv('MEXC_API_SECRET')
        self.base_url = "https://api.mexc.com"

    async def get_top_coins(self, limit):
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/24hr"
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"MEXC API error: {await response.text()}")
                    data = await response.json()
                    sorted_coins = sorted(
                        data,
                        key=lambda x: float(x['quoteVolume']) if x.get('quoteVolume') else 0,
                        reverse=True
                    )
                    top_coins = [coin['symbol'] for coin in sorted_coins if coin['symbol'].endswith('USDT')][:limit]
                    return top_coins
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            async with aiohttp.ClientSession() as session:
                # Klines (1h and 4h)
                klines_1h_url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
                klines_4h_url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval=4h&limit=100"
                ticker_url = f"{self.base_url}/api/v3/ticker/24hr?symbol={symbol}"

                # Fetch 1h klines
                async with session.get(klines_1h_url) as resp_1h:
                    if resp_1h.status != 200:
                        raise Exception(f"Error fetching 1h klines: {await resp_1h.text()}")
                    klines_1h = await resp_1h.json()

                # Fetch 4h klines
                async with session.get(klines_4h_url) as resp_4h:
                    if resp_4h.status != 200:
                        raise Exception(f"Error fetching 4h klines: {await resp_4h.text()}")
                    klines_4h = await resp_4h.json()

                # Fetch ticker
                async with session.get(ticker_url) as resp_ticker:
                    if resp_ticker.status != 200:
                        raise Exception(f"Error fetching ticker: {await resp_ticker.text()}")
                    ticker = await resp_ticker.json()

                return {
                    'coin': symbol,
                    'klines_1h': klines_1h,
                    'klines_4h': klines_4h,
                    'price': float(ticker['lastPrice']),
                    'volume': float(ticker['quoteVolume'])
                }
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None

    async def close(self):
        pass  # aiohttp kullanıldığı için client kapatma gerekmez

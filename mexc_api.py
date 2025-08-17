import aiohttp
import os
import logging
from dotenv import load_dotenv

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                    logger.info(f"Fetched {len(top_coins)} top coins")
                    return top_coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            async with aiohttp.ClientSession() as session:
                # Klines (1h and 4h)
                klines_1h_url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval=60m&limit=100"
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
                    'klines_1h': [[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines_1h],  # [open_time, open, high, low, close, volume]
                    'klines_4h': [[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines_4h],
                    'price': float(ticker['lastPrice']),
                    'volume': float(ticker['quoteVolume'])
                }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def close(self):
        pass  # aiohttp kullanıldığı için client kapatma gerekmez

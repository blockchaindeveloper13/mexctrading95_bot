import os
import json
import logging
from dotenv import load_dotenv

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MEXCClient:
    def __init__(self):
        # API kullanılmayacak, ama yapı korunsun
        self.data_file = os.getenv('MARKET_DATA_FILE', 'market_data.json')

    async def get_top_coins(self, limit):
        try:
            # JSON dosyasından verileri oku
            with open(self.data_file, 'r') as f:
                market_data = json.load(f)
            
            # USDT çiftlerini filtrele
            usdt_pairs = [item['coin'] for item in market_data if item['coin'].endswith('/USDT')]
            logger.info(f"Found {len(usdt_pairs)} USDT pairs (first 5): {usdt_pairs[:5]}...")
            
            # Hacme göre sırala
            sorted_pairs = sorted(
                [(item['coin'], item.get('volume', 0)) for item in market_data if item['coin'] in usdt_pairs],
                key=lambda x: x[1],
                reverse=True
            )
            coins = [symbol for symbol, _ in sorted_pairs[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            if not coins:
                logger.warning("No valid USDT pairs found in data file")
            return coins
        except FileNotFoundError:
            logger.error(f"Market data file not found: {self.data_file}")
            return []
        except Exception as e:
            logger.error(f"Error fetching top coins from data file: {e}")
            return []

    async def get_market_data(self, symbol):
        try:
            # JSON dosyasından verileri oku
            with open(self.data_file, 'r') as f:
                market_data = json.load(f)
            
            # İlgili coin’in verisini bul
            for item in market_data:
                if item['coin'] == symbol:
                    data = {
                        'coin': symbol,
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                        'klines_1h': item.get('klines_1h', []),
                        'klines_4h': item.get('klines_4h', []),
                        'order_book': item.get('order_book', {'bids': [], 'asks': []})
                    }
                    logger.info(f"Fetched market data for {symbol}: price={data['price']}, volume={data['volume']}, klines_1h={len(data['klines_1h'])}, klines_4h={len(data['klines_4h'])}, order_book_bids={len(data['order_book'].get('bids', []))}")
                    return data
            logger.warning(f"No data found for {symbol} in data file")
            return None
        except FileNotFoundError:
            logger.error(f"Market data file not found: {self.data_file}")
            return None
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} from data file: {e}")
            return None

    async def close(self):
        # JSON kullanıldığı için bağlantı kapatma gerekmez
        logger.info("Closing MEXCClient (no-op for JSON data)")

import os
import json
import ccxt.async_support as ccxt
import logging
from dotenv import load_dotenv
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MEXCClient:
    def __init__(self):
        self.exchange = ccxt.mexc3({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultTimeframe': '60m',  # MEXC API v3 için doğru format
                'timeframes': {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '60m',  # 1 saat = 60 dakika
                    '4h': '4h',
                    '1d': '1d',
                }
            }
        })
        self.data_file = os.getenv('MARKET_DATA_FILE', 'market_data.json')

    async def fetch_and_save_market_data(self, symbols, timeframes=['60m', '4h']):  # '1h' yerine '60m' kullanın
        market_data = []
        for symbol in symbols:
            try:
                klines = {}
                for tf in timeframes:
                    try:
                        klines[tf] = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=100)
                    except Exception as e:
                        logger.warning(f"Unsupported timeframe {tf} for symbol {symbol}: {e}")
                        klines[tf] = []
                
                ticker = await self.exchange.fetch_ticker(symbol)
                order_book = await self.exchange.fetch_order_book(symbol, limit=10)
                
                data = {
                    'coin': symbol,
                    'price': ticker.get('last', 0),
                    'volume': ticker.get('quoteVolume', 0),
                    'klines': klines,
                    'order_book': order_book,
                    'timestamp': datetime.utcnow().isoformat()
                }
                market_data.append(data)
                
                # Log bilgilerini daha detaylı hale getirdim
                log_msg = f"Fetched market data for {symbol}: price={data['price']}, volume={data['volume']}"
                for tf in timeframes:
                    log_msg += f", klines_{tf}={len(klines.get(tf, []))}"
                log_msg += f", order_book_bids={len(order_book.get('bids', []))}"
                logger.info(log_msg)
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                continue
        
        # Verileri market_data.json'a kaydet
        try:
            with open(self.data_file, 'w') as f:
                json.dump(market_data, f, indent=2)
            logger.info(f"Saved {len(market_data)} coins to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving market data to {self.data_file}: {e}")
        
        return market_data

    async def get_top_coins(self, limit=10, timeframes=['60m', '4h']):
        try:
            # Market verilerini çek
            markets = await self.exchange.load_markets()
            logger.info(f"Loaded {len(markets)} markets")
            
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
            logger.info(f"Fetched {len(coins)} top coins: {coins}")
            
            if not coins:
                logger.warning("No valid USDT pairs found in tickers")
                return []
            
            # Seçilen coinler için market verilerini çek ve kaydet
            await self.fetch_and_save_market_data(coins, timeframes)
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol, timeframes=['60m', '4h']):
        try:
            # JSON dosyasından verileri oku
            with open(self.data_file, 'r') as f:
                market_data = json.load(f)
            
            # İlgili coin'in verisini bul
            for item in market_data:
                if item['coin'] == symbol:
                    data = {
                        'coin': symbol,
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                        'klines': {tf: item['klines'].get(tf, []) for tf in timeframes},
                        'order_book': item.get('order_book', {'bids': [], 'asks': []})
                    }
                    
                    # Log bilgilerini güncelledim
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
        await self.exchange.close()
        logger.info("Closed MEXCClient connection")

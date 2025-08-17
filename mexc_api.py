import ccxt
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

class MEXCClient:
    def __init__(self):
        self.exchange = ccxt.mexc({
            'apiKey': os.getenv('MEXC_API_KEY'),
            'secret': os.getenv('MEXC_API_SECRET'),
        })

    def get_top_coins(self, limit=100):
        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = {symbol: data for symbol, data in tickers.items() if symbol.endswith('USDT')}
            sorted_pairs = sorted(
                usdt_pairs.items(),
                key=lambda x: float(x[1]['quoteVolume'] or 0),
                reverse=True
            )
            return [pair[0] for pair in sorted_pairs[:limit]]
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return []

    def get_market_data(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            order_book = self.exchange.fetch_order_book(symbol, limit=20)
            klines_1h = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)
            klines_4h = self.exchange.fetch_ohlcv(symbol, timeframe='4h', limit=200)
            return {
                'symbol': symbol,
                'volume_change': ticker.get('percentage', 0),
                'price_change': ticker.get('change', 0),
                'bid_ask_ratio': sum([float(bid[1]) for bid in order_book['bids']]) / 
                                (sum([float(ask[1]) for ask in order_book['asks']]) or 1),
                'klines_1h': klines_1h,
                'klines_4h': klines_4h,
                'last_price': ticker.get('last', 0)
            }
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None

import os
import json
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import logging
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime
import re

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
        # CCXT için zaman aralığı dönüşümü
        tf_mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '1h'  # MEXC API'si 60m yerine 1h kullanıyor
        }
        ccxt_timeframe = tf_mapping.get(timeframe)
        if not ccxt_timeframe:
            logger.error(f"Invalid timeframe for {symbol}: {timeframe}")
            raise ValueError(f"Invalid timeframe: {timeframe}")

        for attempt in range(retries):
            try:
                klines = await self.exchange.fetch_ohlcv(symbol, ccxt_timeframe, limit=limit)
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

    async def fetch_and_save_market_data(self, symbol):
        logger.info(f"Fetching market data for {symbol}")
        timeframes = ['1m', '5m', '15m', '30m', '60m']
        if not timeframes or any(tf is None or tf not in ['1m', '5m', '15m', '30m', '60m'] for tf in timeframes):
            logger.error(f"Invalid timeframes: {timeframes}")
            raise ValueError(f"Invalid timeframes: {timeframes}")

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = float(ticker.get('last', 0)) if ticker.get('last') else 0.0
            volume = float(ticker.get('quoteVolume', 0)) if ticker.get('quoteVolume') else 0.0

            klines = {}
            for tf in timeframes:
                logger.debug(f"Fetching {tf} kline for {symbol}")
                klines[tf] = await self.get_kline(symbol, tf, limit=100, retries=3, delay=2)
                if not klines[tf]:
                    logger.warning(f"No {tf} kline data for {symbol}, skipping")
                    return None
            if not all(klines.get(tf) for tf in timeframes):
                logger.warning(f"Incomplete kline data for {symbol}, skipping")
                return None

            order_book = await self.get_order_book(symbol, limit=10)

            coin_data = {
                'coin': symbol,
                'price': price,
                'volume': volume,
                'klines': klines,
                'order_book': order_book
            }

            # JSON'a kaydet
            try:
                data = []
                if os.path.exists(self.data_file):
                    with open(self.data_file, 'r') as f:
                        data = json.load(f)
                data = [d for d in data if d['coin'] != symbol]  # Eski veriyi güncelle
                data.append(coin_data)
                with open(self.data_file, 'w') as f:
                    json.dump(data, f)
                logger.info(f"Saved data for {symbol} to {self.data_file}")
            except Exception as e:
                logger.error(f"Error saving to {self.data_file}: {e}")
                return None

            logger.info(f"Data for {symbol}: price={price}, volume={volume}, "
                       f"klines_1m={len(klines.get('1m', []))}, klines_5m={len(klines.get('5m', []))}, "
                       f"klines_15m={len(klines.get('15m', []))}, klines_30m={len(klines.get('30m', []))}, "
                       f"klines_60m={len(klines.get('60m', []))}")
            await asyncio.sleep(2.0)  # Rate limit için
            return coin_data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def get_top_coins(self, limit=100):
        logger.debug(f"Fetching top {limit} coins")
        try:
            markets = await self.exchange.load_markets()
            usdt_pairs = [s for s in markets if s.endswith('/USDT')]
            logger.info(f"{len(usdt_pairs)} USDT pairs found: {usdt_pairs[:5]}...")
            tickers = await self.get_tickers()
            logger.info(f"Fetched {len(tickers)} tickers")
            sorted_tickers = sorted(
                [(s, tickers[s].get('quoteVolume', 0)) for s in usdt_pairs if s in tickers],
                key=lambda x: x[1], reverse=True
            )
            coins = [s for s, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            if not coins:
                logger.warning("No valid USDT pairs found")
                return []
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol, timeframes=['1m', '5m', '15m', '30m', '60m']):
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
                               f"klines_1m={len(data['klines']['1m'])}, klines_5m={len(data['klines']['5m'])}, "
                               f"klines_15m={len(data['klines']['15m'])}, klines_30m={len(data['klines']['30m'])}, "
                               f"klines_60m={len(data['klines']['60m'])}")
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

def calculate_indicators(klines_1m, klines_5m, klines_15m, klines_30m, klines_60m, order_book=None):
    logger.debug(f"Calculating indicators: klines_1m={len(klines_1m)}, klines_5m={len(klines_5m)}, "
                f"klines_15m={len(klines_15m)}, klines_30m={len(klines_30m)}, klines_60m={len(klines_60m)}")
    try:
        if not all([klines_1m, klines_5m, klines_15m, klines_30m, klines_60m]):
            logger.warning("Empty klines data provided")
            return None

        # DataFrame'leri oluştur
        dfs = {}
        for tf, klines in zip(['1m', '5m', '15m', '30m', '60m'], [klines_1m, klines_5m, klines_15m, klines_30m, klines_60m]):
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            dfs[tf] = df

        indicators = {}
        for tf in ['1m', '5m', '15m', '30m', '60m']:
            df = dfs[tf]
            indicators[f'rsi_{tf}'] = ta.rsi(df['close'], length=14).iloc[-1] if len(df) >= 14 else None
            indicators[f'ema_20_{tf}'] = ta.ema(df['close'], length=20).iloc[-1] if len(df) >= 20 else None
            indicators[f'ema_50_{tf}'] = ta.ema(df['close'], length=50).iloc[-1] if len(df) >= 50 else None
            indicators[f'macd_{tf}'] = ta.macd(df['close'])['MACD_12_26_9'].iloc[-1] if len(df) >= 26 else None
            indicators[f'macd_signal_{tf}'] = ta.macd(df['close'])['MACDs_12_26_9'].iloc[-1] if len(df) >= 26 else None
            indicators[f'bb_upper_{tf}'] = ta.bbands(df['close'], length=20)['BBU_20_2.0'].iloc[-1] if len(df) >= 20 else None
            indicators[f'bb_lower_{tf}'] = ta.bbands(df['close'], length=20)['BBL_20_2.0'].iloc[-1] if len(df) >= 20 else None
            indicators[f'vwap_{tf}'] = ta.vwap(df['high'], df['low'], df['close'], df['volume']).iloc[-1] if len(df) >= 1 else None
            indicators[f'stoch_k_{tf}'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3'].iloc[-1] if len(df) >= 14 else None
            indicators[f'stoch_d_{tf}'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHd_14_3_3'].iloc[-1] if len(df) >= 14 else None
            indicators[f'atr_{tf}'] = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else None

            if len(df) >= 2:
                indicators[f'volume_change_{tf}'] = ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] * 100) if df['volume'].iloc[-2] != 0 else None
            else:
                indicators[f'volume_change_{tf}'] = None

        if order_book:
            bids = sum(float(bid[1]) for bid in order_book.get('bids', [])[:10])
            asks = sum(float(ask[1]) for ask in order_book.get('asks', [])[:10])
            indicators['bid_ask_ratio'] = (bids / asks) if asks != 0 else None

        logger.info("Indicators calculated successfully")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

class DeepSeekClient:
    def __init__(self):
        logger.debug("Initializing DeepSeekClient")
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def clean_json_response(self, response):
        logger.debug(f"Cleaning JSON response: {response[:200]}...")
        try:
            cleaned = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE)
            cleaned = re.sub(r'\n\s*###.*|\n\s*\*.*', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return response

    def analyze_coin(self, data, trade_type='spot'):
        logger.debug(f"Analyzing coin: {data.get('coin', 'Unknown')} ({trade_type})")
        try:
            symbol = data.get('coin', 'Unknown')
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            indicators = data.get('indicators', {})

            prompt = (
                f"Analyze {symbol} ({trade_type.upper()} trading):\n"
                f"Price: ${price}\nVolume: ${volume}\n"
                f"1m RSI: {indicators.get('rsi_1m', 'N/A')}\n"
                f"1m EMA20: {indicators.get('ema_20_1m', 'N/A')}\n"
                f"1m EMA50: {indicators.get('ema_50_1m', 'N/A')}\n"
                f"1m MACD: {indicators.get('macd_1m', 'N/A')}\n"
                f"1m MACD Signal: {indicators.get('macd_signal_1m', 'N/A')}\n"
                f"1m Bollinger Upper: {indicators.get('bb_upper_1m', 'N/A')}\n"
                f"1m Bollinger Lower: {indicators.get('bb_lower_1m', 'N/A')}\n"
                f"1m VWAP: {indicators.get('vwap_1m', 'N/A')}\n"
                f"1m Stochastic %K: {indicators.get('stoch_k_1m', 'N/A')}\n"
                f"1m Stochastic %D: {indicators.get('stoch_d_1m', 'N/A')}\n"
                f"1m ATR: {indicators.get('atr_1m', 'N/A')}\n"
                f"5m RSI: {indicators.get('rsi_5m', 'N/A')}\n"
                f"5m EMA20: {indicators.get('ema_20_5m', 'N/A')}\n"
                f"5m EMA50: {indicators.get('ema_50_5m', 'N/A')}\n"
                f"5m MACD: {indicators.get('macd_5m', 'N/A')}\n"
                f"5m MACD Signal: {indicators.get('macd_signal_5m', 'N/A')}\n"
                f"5m Bollinger Upper: {indicators.get('bb_upper_5m', 'N/A')}\n"
                f"5m Bollinger Lower: {indicators.get('bb_lower_5m', 'N/A')}\n"
                f"5m VWAP: {indicators.get('vwap_5m', 'N/A')}\n"
                f"5m Stochastic %K: {indicators.get('stoch_k_5m', 'N/A')}\n"
                f"5m Stochastic %D: {indicators.get('stoch_d_5m', 'N/A')}\n"
                f"5m ATR: {indicators.get('atr_5m', 'N/A')}\n"
                f"15m RSI: {indicators.get('rsi_15m', 'N/A')}\n"
                f"15m EMA20: {indicators.get('ema_20_15m', 'N/A')}\n"
                f"15m EMA50: {indicators.get('ema_50_15m', 'N/A')}\n"
                f"15m MACD: {indicators.get('macd_15m', 'N/A')}\n"
                f"15m MACD Signal: {indicators.get('macd_signal_15m', 'N/A')}\n"
                f"15m Bollinger Upper: {indicators.get('bb_upper_15m', 'N/A')}\n"
                f"15m Bollinger Lower: {indicators.get('bb_lower_15m', 'N/A')}\n"
                f"15m VWAP: {indicators.get('vwap_15m', 'N/A')}\n"
                f"15m Stochastic %K: {indicators.get('stoch_k_15m', 'N/A')}\n"
                f"15m Stochastic %D: {indicators.get('stoch_d_15m', 'N/A')}\n"
                f"15m ATR: {indicators.get('atr_15m', 'N/A')}\n"
                f"30m RSI: {indicators.get('rsi_30m', 'N/A')}\n"
                f"30m EMA20: {indicators.get('ema_20_30m', 'N/A')}\n"
                f"30m EMA50: {indicators.get('ema_50_30m', 'N/A')}\n"
                f"30m MACD: {indicators.get('macd_30m', 'N/A')}\n"
                f"30m MACD Signal: {indicators.get('macd_signal_30m', 'N/A')}\n"
                f"30m Bollinger Upper: {indicators.get('bb_upper_30m', 'N/A')}\n"
                f"30m Bollinger Lower: {indicators.get('bb_lower_30m', 'N/A')}\n"
                f"30m VWAP: {indicators.get('vwap_30m', 'N/A')}\n"
                f"30m Stochastic %K: {indicators.get('stoch_k_30m', 'N/A')}\n"
                f"30m Stochastic %D: {indicators.get('stoch_d_30m', 'N/A')}\n"
                f"30m ATR: {indicators.get('atr_30m', 'N/A')}\n"
                f"60m RSI: {indicators.get('rsi_60m', 'N/A')}\n"
                f"60m EMA20: {indicators.get('ema_20_60m', 'N/A')}\n"
                f"60m EMA50: {indicators.get('ema_50_60m', 'N/A')}\n"
                f"60m MACD: {indicators.get('macd_60m', 'N/A')}\n"
                f"60m MACD Signal: {indicators.get('macd_signal_60m', 'N/A')}\n"
                f"60m Bollinger Upper: {indicators.get('bb_upper_60m', 'N/A')}\n"
                f"60m Bollinger Lower: {indicators.get('bb_lower_60m', 'N/A')}\n"
                f"60m VWAP: {indicators.get('vwap_60m', 'N/A')}\n"
                f"60m Stochastic %K: {indicators.get('stoch_k_60m', 'N/A')}\n"
                f"60m Stochastic %D: {indicators.get('stoch_d_60m', 'N/A')}\n"
                f"60m ATR: {indicators.get('atr_60m', 'N/A')}\n"
                f"Volume Change 1m: {indicators.get('volume_change_1m', 'N/A')}%;\n"
                f"Volume Change 5m: {indicators.get('volume_change_5m', 'N/A')}%;\n"
                f"Volume Change 15m: {indicators.get('volume_change_15m', 'N/A')}%;\n"
                f"Volume Change 30m: {indicators.get('volume_change_30m', 'N/A')}%;\n"
                f"Volume Change 60m: {indicators.get('volume_change_60m', 'N/A')}%;\n"
                f"Bid/Ask Ratio: {indicators.get('bid_ask_ratio', 'N/A')}\n"
                "Provide short-term trading analysis (1-4 hours). Return JSON with:\n"
                "- pump_probability (0-100%)\n- dump_probability (0-100%)\n- entry_price\n- exit_price\n- stop_loss\n- leverage\n- fundamental_analysis\n"
                "Return only JSON, no extra text."
            )

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response.choices[0].message.content
            cleaned = self.clean_json_response(analysis)
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    logger.warning(f"Invalid JSON for {symbol}: {cleaned}")
                    return {'short_term': {'pump_probability': 0, 'dump_probability': 0, 'entry_price': price, 'exit_price': price, 'stop_loss': price * 0.95, 'leverage': 'N/A', 'fundamental_analysis': 'No data'}}
                return {'short_term': parsed}
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error for {symbol}: {cleaned}")
                return {'short_term': {'pump_probability': 0, 'dump_probability': 0, 'entry_price': price, 'exit_price': price, 'stop_loss': price * 0.95, 'leverage': 'N/A', 'fundamental_analysis': 'No data'}}
        except Exception as e:
            logger.error(f"Error analyzing {symbol} ({trade_type}): {e}")
            return {'short_term': {'pump_probability': 0, 'dump_probability': 0, 'entry_price': price, 'exit_price': price, 'stop_loss': price * 0.95, 'leverage': 'N/A', 'fundamental_analysis': 'No data'}}

class Storage:
    def save_analysis(self, data):
        logger.debug(f"Saving analysis data")
        try:
            if not data:
                logger.warning("No valid data to save")
                return
            with open('/tmp/analysis.json', 'w') as f:
                json.dump(data, f, indent=4)
            logger.info("Analysis saved to /tmp/analysis.json")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    def load_analysis(self):
        logger.debug("Loading analysis from /tmp/analysis.json")
        try:
            with open('/tmp/analysis.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return {}

class TelegramBot:
    def __init__(self):
        logger.debug("Initializing TelegramBot")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        self.app = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
        self.web_app = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug(f"Start command from chat_id={update.effective_chat.id}")
        keyboard = [
            [InlineKeyboardButton("Top 100 Spot Analizi", callback_data='top_100_spot')],
            [InlineKeyboardButton("Top 300 Spot Analizi", callback_data='top_300_spot')],
            [InlineKeyboardButton("Top 100 Vadeli Analizi", callback_data='top_100_futures')],
            [InlineKeyboardButton("Top 300 Vadeli Analizi", callback_data='top_300_futures')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Analiz için butonları kullanabilirsiniz:", reply_markup=reply_markup)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        logger.debug(f"Button clicked: {query.data}")
        try:
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            await query.message.reply_text(f"{trade_type.upper()} analizi yapılıyor (Top {limit})...")
            context.job_queue.run_once(
                self.analyze_and_send,
                0,
                data={'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type},
                chat_id=self.group_id
            )
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.message.reply_text(f"Error: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE):
        data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.debug(f"Analyzing for chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"No significant results for Top {limit} {trade_type} analysis.")
            logger.info(f"Analysis completed for Top {limit} {trade_type}")
        except Exception as e:
            logger.error(f"Error in analyze_and_send: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"Error during {trade_type} analysis: {str(e)}")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug(f"Chat message: {update.message.text}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": update.message.text}]
                ),
                timeout=20
            )
            await update.message.reply_text(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(f"Error: {str(e)}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug("Show analysis command")
        try:
            storage = Storage()
            data = storage.load_analysis()
            message_spot = self.format_results(data, 'spot')
            message_futures = self.format_results(data, 'futures')
            messages = []
            if not message_spot.strip().endswith("TOP_100_SPOT:\n") and not message_spot.strip().endswith("TOP_300_SPOT:\n"):
                messages.append(message_spot)
            if not message_futures.strip().endswith("TOP_100_FUTURES:\n") and not message_futures.strip().endswith("TOP_300_FUTURES:\n"):
                messages.append(message_futures)
            if messages:
                await update.message.reply_text("\n\n".join(messages))
            else:
                await update.message.reply_text("No analysis results found.")
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            await update.message.reply_text(f"Error: {str(e)}")

    def format_results(self, coin_data, trade_type, symbol):
        logger.debug(f"Formatting results for {symbol} ({trade_type})")
        indicators = coin_data.get('indicators', {})
        analysis = coin_data.get('deepseek_analysis', {}).get('short_term', {})
        message = (
            f"📊 {symbol} {trade_type.upper()} Analizi ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            f"- Kısa Vadeli: Giriş: ${analysis.get('entry_price', 0):.2f} | "
            f"Çıkış: ${analysis.get('exit_price', 0):.2f} | "
            f"Stop Loss: ${analysis.get('stop_loss', 0):.2f} | "
            f"Kaldıraç: {analysis.get('leverage', 'N/A')}\n"
            f"- Pump Olasılığı: {analysis.get('pump_probability', 0)}% | "
            f"Dump Olasılığı: {analysis.get('dump_probability', 0)}%\n"
            f"- Temel Analiz: {analysis.get('fundamental_analysis', 'No data')}\n"
            f"- Hacim Değişimleri: 1m: {indicators.get('volume_change_1m', 'N/A'):.2f}% | "
            f"5m: {indicators.get('volume_change_5m', 'N/A'):.2f}% | "
            f"15m: {indicators.get('volume_change_15m', 'N/A'):.2f}% | "
            f"30m: {indicators.get('volume_change_30m', 'N/A'):.2f}% | "
            f"60m: {indicators.get('volume_change_60m', 'N/A'):.2f}%\n"
            f"- Bid/Ask Oranı: {indicators.get('bid_ask_ratio', 'N/A'):.2f}\n"
        )
        return message

    async def process_coin(self, symbol, mexc, deepseek, trade_type, chat_id):
        logger.debug(f"Processing coin: {symbol} ({trade_type})")
        try:
            # Veriyi çek ve JSON'a kaydet
            data = await mexc.fetch_and_save_market_data(symbol)
            if not data or not all(data['klines'].get(tf) for tf in ['1m', '5m', '15m', '30m', '60m']):
                logger.warning(f"No valid market data for {symbol} ({trade_type})")
                return None

            # Göstergeleri hesapla
            data['indicators'] = calculate_indicators(
                data['klines']['1m'], data['klines']['5m'], data['klines']['15m'],
                data['klines']['30m'], data['klines']['60m'], data.get('order_book')
            )
            if not data['indicators']:
                logger.warning(f"No indicators for {symbol} ({trade_type})")
                return None

            # DeepSeek analizi yap
            data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
            logger.info(f"Processed {symbol} ({trade_type}): price={data.get('price')}, "
                       f"klines_60m={len(data['klines']['60m'])}")

            # Telegram'a gönder
            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Analysis sent for {symbol} ({trade_type})")

            # Analizi kaydet
            storage = Storage()
            storage.save_analysis({f'{symbol}_{trade_type}': [data]})

            return data
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"Error processing {symbol}: {str(e)}")
            return None

    async def analyze_coins(self, limit, trade_type, chat_id):
        logger.debug(f"Starting analyze_coins for limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        deepseek = DeepSeekClient()

        coins = await mexc.get_top_coins(limit)
        logger.info(f"Analyzing {len(coins)} coins: {coins[:5]}...")

        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_{limit}_{trade_type}': []}
        for symbol in coins:
            coin_data = await self.process_coin(symbol, mexc, deepseek, trade_type, chat_id)
            if coin_data:
                results[f'top_{limit}_{trade_type}'].append(coin_data)
            await asyncio.sleep(2.0)  # Coin'ler arasında rate limit için bekleme

        logger.info(f"Processed {len(results[f'top_{limit}_{trade_type}'])} valid coins for Top {limit} {trade_type}")
        await mexc.close()
        return results

    async def webhook_handler(self, request):
        logger.debug("Webhook request received")
        try:
            raw_data = await request.json()
            update = Update.de_json(raw_data, self.app.bot)
            if not update:
                logger.warning("Invalid webhook update")
                return web.Response(text="ERROR: Invalid update", status=400)
            await self.app.process_update(update)
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(text=f"ERROR: {str(e)}", status=500)

    async def run(self):
        logger.debug("Starting webhook server")
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        await self.app.bot.set_webhook(url=webhook_url)
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook server running on port {os.getenv('PORT', 8443)}")
        await asyncio.Event().wait()

def main():
    logger.debug("Starting main")
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    logger.debug("Script started")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_coins":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        trade_type = sys.argv[3] if len(sys.argv) > 3 else 'spot'
        bot = TelegramBot()
        asyncio.run(bot.analyze_coins(limit, trade_type, bot.group_id))
    else:
        main()

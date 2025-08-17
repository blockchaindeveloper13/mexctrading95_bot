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
            if not isinstance(timeframe, str) or timeframe not in ['1m', '5m', '15m', '30m', '60m', '4h', '1d']:
                logger.error(f"Invalid or None timeframe for {symbol}: {timeframe}")
                raise ValueError(f"Invalid or None timeframe: {timeframe}")
            for attempt in range(retries):
                try:
                    klines = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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

    async def fetch_and_save_market_data(self, symbols=None, timeframes=['60m', '4h']):
        logger.info("Starting fetch_and_save_market_data")
        if not timeframes or any(tf is None or tf not in ['1m', '5m', '15m', '30m', '60m', '4h', '1d'] for tf in timeframes):
            logger.error(f"Invalid timeframes: {timeframes}")
            raise ValueError(f"Invalid timeframes: {timeframes}")
        
        markets = await self.get_exchange_info()
        usdt_pairs = [m['symbol'] for m in markets['symbols'] if m['symbol'].endswith('USDT')]
        Cann logger.info(f"{len(usdt_pairs)} USDT pairs found: {usdt_pairs[:5]}...")

        tickers = await self.get_tickers()
        logger.info(f"Fetched {len(tickers)} tickers")

        top_symbols = sorted(
            [(s, tickers[s].get('quoteVolume', 0)) for s in usdt_pairs if s in tickers],
            key=lambda x: x[1], reverse=True
        )
        top_symbols = [s for s, _ in top_symbols]
        if symbols:
            top_symbols = [s for s in top_symbols if s in symbols]
        else:
            top_symbols = top_symbols[:100]
        logger.info(f"Top {len(top_symbols)} symbols: {top_symbols[:5]}...")

        data = []
        for symbol in top_symbols:
            try:
                ticker = tickers.get(symbol, {})
                price = float(ticker.get('lastPrice', 0)) if ticker.get('lastPrice') else 0.0
                volume = float(ticker.get('quoteVolume', 0)) if ticker.get('quoteVolume') else 0.0

                klines = {}
                for tf in timeframes:
                    logger.debug(f"Fetching {tf} kline for {symbol}")
                    klines[tf] = await self.get_kline(symbol, tf, limit=100, retries=3, delay=2)
                    if not klines[tf]:
                        logger.warning(f"No {tf} kline data for {symbol}, skipping")
                        break
                if not all(klines.get(tf) for tf in timeframes):
                    logger.warning(f"Incomplete kline data for {symbol}, skipping")
                    continue

                order_book = await self.get_order_book(symbol, limit=10)

                coin_data = {
                    'coin': symbol,
                    'price': price,
                    'volume': volume,
                    'klines': klines,
                    'order_book': order_book
                }
                data.append(coin_data)
                logger.info(f"Data for {symbol}: price={price}, volume={volume}, "
                           f"klines_60m={len(klines.get('60m', []))}, klines_4h={len(klines.get('4h', []))}")

                await asyncio.sleep(2.0)  # Rate limit için artırıldı
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(data)} coins to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving to {self.data_file}: {e}")

        return data

    async def get_top_coins(self, limit=100, timeframes=['60m', '4h']):
        logger.debug(f"Fetching top {limit} coins")
        try:
            markets = await self.exchange.load_markets()
            usdt_pairs = [s for s in markets if s.endswith('/USDT')]
            tickers = await self.get_tickers()
            sorted_tickers = sorted(
                [(s, tickers[s].get('quoteVolume', 0)) for s in usdt_pairs if s in tickers],
                key=lambda x: x[1], reverse=True
            )
            coins = [s for s, _ in sorted_tickers[:limit]]
            logger.info(f"Fetched {len(coins)} top coins: {coins[:5]}...")
            if not coins:
                logger.warning("No valid USDT pairs found")
                return []
            await self.fetch_and_save_market_data(coins, timeframes)
            return coins
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def get_market_data(self, symbol, timeframes=['60m', '4h']):
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
                               f"klines_60m={len(data['klines']['60m'])}, klines_4h={len(data['klines']['4h'])}")
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

def calculate_indicators(klines_60m, klines_4h, order_book=None):
    logger.debug(f"Calculating indicators: klines_60m length={len(klines_60m)}, klines_4h length={len(klines_4h)}")
    try:
        if not klines_60m or not klines_4h:
            logger.warning("Empty klines data provided")
            return None

        df_60m = pd.DataFrame(klines_60m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_60m['timestamp'] = pd.to_datetime(df_60m['timestamp'], unit='ms')
        df_60m.set_index('timestamp', inplace=True)
        df_60m = df_60m.astype(float)

        df_4h = pd.DataFrame(klines_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
        df_4h.set_index('timestamp', inplace=True)
        df_4h = df_4h.astype(float)

        indicators = {}
        indicators['rsi_60m'] = ta.rsi(df_60m['close'], length=14).iloc[-1] if len(df_60m) >= 14 else None
        indicators['ema_20_60m'] = ta.ema(df_60m['close'], length=20).iloc[-1] if len(df_60m) >= 20 else None
        indicators['ema_50_60m'] = ta.ema(df_60m['close'], length=50).iloc[-1] if len(df_60m) >= 50 else None
        indicators['macd_60m'] = ta.macd(df_60m['close'])['MACD_12_26_9'].iloc[-1] if len(df_60m) >= 26 else None
        indicators['macd_signal_60m'] = ta.macd(df_60m['close'])['MACDs_12_26_9'].iloc[-1] if len(df_60m) >= 26 else None
        indicators['bb_upper_60m'] = ta.bbands(df_60m['close'], length=20)['BBU_20_2.0'].iloc[-1] if len(df_60m) >= 20 else None
        indicators['bb_lower_60m'] = ta.bbands(df_60m['close'], length=20)['BBL_20_2.0'].iloc[-1] if len(df_60m) >= 20 else None
        indicators['vwap_60m'] = ta.vwap(df_60m['high'], df_60m['low'], df_60m['close'], df_60m['volume']).iloc[-1] if len(df_60m) >= 1 else None
        indicators['stoch_k_60m'] = ta.stoch(df_60m['high'], df_60m['low'], df_60m['close'])['STOCHk_14_3_3'].iloc[-1] if len(df_60m) >= 14 else None
        indicators['stoch_d_60m'] = ta.stoch(df_60m['high'], df_60m['low'], df_60m['close'])['STOCHd_14_3_3'].iloc[-1] if len(df_60m) >= 14 else None
        indicators['atr_60m'] = ta.atr(df_60m['high'], df_60m['low'], df_60m['close'], length=14).iloc[-1] if len(df_60m) >= 14 else None

        indicators['rsi_4h'] = ta.rsi(df_4h['close'], length=14).iloc[-1] if len(df_4h) >= 14 else None
        indicators['ema_20_4h'] = ta.ema(df_4h['close'], length=20).iloc[-1] if len(df_4h) >= 20 else None
        indicators['ema_50_4h'] = ta.ema(df_4h['close'], length=50).iloc[-1] if len(df_4h) >= 50 else None
        indicators['macd_4h'] = ta.macd(df_4h['close'])['MACD_12_26_9'].iloc[-1] if len(df_4h) >= 26 else None
        indicators['macd_signal_4h'] = ta.macd(df_4h['close'])['MACDs_12_26_9'].iloc[-1] if len(df_4h) >= 26 else None
        indicators['bb_upper_4h'] = ta.bbands(df_4h['close'], length=20)['BBU_20_2.0'].iloc[-1] if len(df_4h) >= 20 else None
        indicators['bb_lower_4h'] = ta.bbands(df_4h['close'], length=20)['BBL_20_2.0'].iloc[-1] if len(df_4h) >= 20 else None
        indicators['vwap_4h'] = ta.vwap(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume']).iloc[-1] if len(df_4h) >= 1 else None
        indicators['stoch_k_4h'] = ta.stoch(df_4h['high'], df_4h['low'], df_4h['close'])['STOCHk_14_3_3'].iloc[-1] if len(df_4h) >= 14 else None
        indicators['stoch_d_4h'] = ta.stoch(df_4h['high'], df_4h['low'], df_4h['close'])['STOCHd_14_3_3'].iloc[-1] if len(df_4h) >= 14 else None
        indicators['atr_4h'] = ta.atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14).iloc[-1] if len(df_4h) >= 14 else None

        if len(df_60m) >= 2:
            indicators['volume_change_60m'] = ((df_60m['volume'].iloc[-1] - df_60m['volume'].iloc[-2]) / df_60m['volume'].iloc[-2] * 100) if df_60m['volume'].iloc[-2] != 0 else None
        else:
            indicators['volume_change_60m'] = None

        if len(df_60m) >= 6:
            last_3h_volume = df_60m['volume'].iloc[-3:].sum()
            prev_3h_volume = df_60m['volume'].iloc[-6:-3].sum()
            indicators['volume_change_3h'] = ((last_3h_volume - prev_3h_volume) / prev_3h_volume * 100) if prev_3h_volume != 0 else None
        else:
            indicators['volume_change_3h'] = None

        if len(df_60m) >= 12:
            last_6h_volume = df_60m['volume'].iloc[-6:].sum()
            prev_6h_volume = df_60m['volume'].iloc[-12:-6].sum()
            indicators['volume_change_6h'] = ((last_6h_volume - prev_6h_volume) / prev_6h_volume * 100) if prev_6h_volume != 0 else None
        else:
            indicators['volume_change_6h'] = None

        if len(df_60m) >= 48:
            last_24h_volume = df_60m['volume'].iloc[-24:].sum()
            prev_24h_volume = df_60m['volume'].iloc[-48:-24].sum()
            indicators['volume_change_24h'] = ((last_24h_volume - prev_24h_volume) / prev_24h_volume * 100) if prev_24h_volume != 0 else None
        else:
            indicators['volume_change_24h'] = None

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
                f"4h RSI: {indicators.get('rsi_4h', 'N/A')}\n"
                f"4h EMA20: {indicators.get('ema_20_4h', 'N/A')}\n"
                f"4h EMA50: {indicators.get('ema_50_4h', 'N/A')}\n"
                f"4h MACD: {indicators.get('macd_4h', 'N/A')}\n"
                f"4h MACD Signal: {indicators.get('macd_signal_4h', 'N/A')}\n"
                f"4h Bollinger Upper: {indicators.get('bb_upper_4h', 'N/A')}\n"
                f"4h Bollinger Lower: {indicators.get('bb_lower_4h', 'N/A')}\n"
                f"4h VWAP: {indicators.get('vwap_4h', 'N/A')}\n"
                f"4h Stochastic %K: {indicators.get('stoch_k_4h', 'N/A')}\n"
                f"4h Stochastic %D: {indicators.get('stoch_d_4h', 'N/A')}\n"
                f"4h ATR: {indicators.get('atr_4h', 'N/A')}\n"
                f"Volume Change 60m: {indicators.get('volume_change_60m', 'N/A')}%;\n"
                f"Volume Change 3h: {indicators.get('volume_change_3h', 'N/A')}%;\n"
                f"Volume Change 6h: {indicators.get('volume_change_6h', 'N/A')}%;\n"
                f"Volume Change 24h: {indicators.get('volume_change_24h', 'N/A')}%;\n"
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
            if not any(data.get(key, []) for key in data if key.startswith('top_')):
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
            results = await self.analyze_coins(limit, trade_type)
            message = self.format_results(results, trade_type)
            if message.strip().endswith(f"TOP_{limit}_{trade_type.upper()}:\n"):
                message = f"No significant results for Top {limit} {trade_type} analysis."
            await context.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Analysis sent for Top {limit} {trade_type}")
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

    def format_results(self, data, trade_type):
        logger.debug(f"Formatting results for {trade_type}")
        message = f"📊 {trade_type.upper()} Günlük Coin Analizi ({data.get('date', 'Unknown')})\n"
        for category in [f'top_100_{trade_type}', f'top_300_{trade_type}']:
            message += f"\n{category.upper()}:\n"
            coins = data.get(category, [])
            if not coins:
                message += "No coins analyzed yet.\n"
                continue
            coins.sort(key=lambda x: x.get('deepseek_analysis', {}).get('short_term', {}).get('pump_probability', 0), reverse=True)
            for i, coin in enumerate(coins[:10], 1):
                indicators = coin.get('indicators', {})
                message += f"{i}. {coin.get('coin', 'Unknown')}\n"
                message += (
                    f"- Kısa Vadeli: Giriş: ${coin.get('deepseek_analysis', {}).get('short_term', {}).get('entry_price', 0):.2f} | "
                    f"Çıkış: ${coin.get('deepseek_analysis', {}).get('short_term', {}).get('exit_price', 0):.2f} | "
                    f"Stop Loss: ${coin.get('deepseek_analysis', {}).get('short_term', {}).get('stop_loss', 0):.2f} | "
                    f"Kaldıraç: {coin.get('deepseek_analysis', {}).get('short_term', {}).get('leverage', 'N/A')}\n"
                    f"- Pump Olasılığı: {coin.get('deepseek_analysis', {}).get('short_term', {}).get('pump_probability', 0)}% | "
                    f"Dump Olasılığı: {coin.get('deepseek_analysis', {}).get('short_term', {}).get('dump_probability', 0)}%\n"
                    f"- Temel Analiz: {coin.get('deepseek_analysis', {}).get('short_term', {}).get('fundamental_analysis', 'No data')}\n"
                    f"- Hacim Değişimleri: 60m: {indicators.get('volume_change_60m', 'N/A'):.2f}% | "
                    f"3h: {indicators.get('volume_change_3h', 'N/A'):.2f}% | "
                    f"6h: {indicators.get('volume_change_6h', 'N/A'):.2f}% | "
                    f"24h: {indicators.get('volume_change_24h', 'N/A'):.2f}%\n"
                    f"- Bid/Ask Oranı: {indicators.get('bid_ask_ratio', 'N/A'):.2f}\n"
                )
        return message

    async def process_coin(self, symbol, mexc, deepseek, trade_type):
        logger.debug(f"Processing coin: {symbol} ({trade_type})")
        try:
            data = await mexc.get_market_data(symbol)
            if not data or not data['klines']['60m'] or not data['klines']['4h']:
                logger.warning(f"No valid market data for {symbol} ({trade_type})")
                return None
            data['indicators'] = calculate_indicators(data['klines']['60m'], data['klines']['4h'], data.get('order_book'))
            if not data['indicators']:
                logger.warning(f"No indicators for {symbol} ({trade_type})")
                return None
            data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
            logger.info(f"Processed {symbol} ({trade_type}): price={data.get('price')}, klines_60m={len(data['klines']['60m'])}")
            return data
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            return None

    async def analyze_coins(self, limit, trade_type):
        logger.debug(f"Starting analyze_coins for limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        deepseek = DeepSeekClient()
        storage = Storage()

        coins = await mexc.get_top_coins(limit)
        logger.info(f"Analyzing {len(coins)} coins: {coins[:5]}...")

        tasks = [self.process_coin(symbol, mexc, deepseek, trade_type) for symbol in coins]
        coin_data = await asyncio.gather(*tasks, return_exceptions=True)

        valid_coins = []
        for symbol, data in zip(coins, coin_data):
            if data and not isinstance(data, Exception):
                valid_coins.append(data)
            else:
                logger.warning(f"No data for {symbol}")

        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_100_{trade_type}': [], f'top_300_{trade_type}': []}
        results[f'top_{limit}_{trade_type}'] = valid_coins
        logger.info(f"Processed {len(valid_coins)} valid coins for Top {limit} {trade_type}")

        storage.save_analysis(results)
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
        asyncio.run(bot.analyze_coins(limit, trade_type))
    else:
        main()

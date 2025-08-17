import os
import json
import pandas as pd
import pandas_ta as ta
import logging
import asyncio
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime
import re

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# GitHub endpoints.json URL
ENDPOINTS_JSON_URL = "https://raw.githubusercontent.com/blockchaindeveloper13/mexctrading95_bot/main/endpoints.json"

class MEXCClient:
    """Handles MEXC API communication."""
    def __init__(self):
        self.base_url = "https://api.mexc.com"

    async def fetch_and_save_market_data(self, symbol, endpoint=None):
        """Fetches market data for the specified symbol."""
        logger.info(f"Fetching market data for {symbol}")
        try:
            async with aiohttp.ClientSession() as session:
                klines = {}
                if endpoint:
                    async with session.get(endpoint) as response:
                        if response.status == 200:
                            klines['60m'] = await response.json()
                        else:
                            logger.warning(f"Failed to fetch 60m kline data for {symbol}: {response.status}")
                            klines['60m'] = []
                else:
                    url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval=60m&limit=100"
                    async with session.get(url) as response:
                        if response.status == 200:
                            klines['60m'] = await response.json()
                        else:
                            logger.warning(f"Failed to fetch 60m kline data for {symbol}: {response.status}")
                            klines['60m'] = []
                await asyncio.sleep(0.5)

                order_book_url = f"{self.base_url}/api/v3/depth?symbol={symbol}&limit=10"
                async with session.get(order_book_url) as order_book_response:
                    order_book = await order_book_response.json() if order_book_response.status == 200 else {}
                await asyncio.sleep(0.5)

                ticker_url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(ticker_url) as ticker_response:
                    ticker = await ticker_response.json() if ticker_response.status == 200 else {'price': '0.0'}
                await asyncio.sleep(0.5)

                return {
                    'klines': klines,
                    'order_book': order_book,
                    'price': float(ticker.get('price', 0.0))
                }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def validate_symbol(self, symbol):
        """Validates if the symbol is valid."""
        logger.info(f"Validating symbol {symbol}")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error validating {symbol}: {e}")
            return False

    async def get_top_coins(self, limit):
        """Fetches top coins by volume."""
        logger.info(f"Fetching top {limit} coins")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/24hr"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        usdt_pairs = [coin for coin in data if coin['symbol'].endswith('USDT')]
                        sorted_coins = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                        return [coin['symbol'] for coin in sorted_coins[:limit]]
                    else:
                        logger.warning("Failed to fetch top coins")
                        return []
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            return []

    async def close(self):
        """Closes MEXC API connection."""
        logger.info("Closing MEXCClient connection")
        pass

class DeepSeekClient:
    """Handles DeepSeek API for coin analysis."""
    def __init__(self):
        self.client = Open

AI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    self.storage = Storage()

    def analyze_coin(self, symbol, data, trade_type, chat_id):
        """Performs coin analysis using DeepSeek API."""
        logger.info(f"Analyzing {symbol} for {trade_type}, chat_id={chat_id}")
        try:
            conversations = self.storage.load_conversations()
            group_context = conversations.get(str(chat_id), [])
            context_str = "\n".join([f"[{c['timestamp']}] {c['message']}" for c in group_context])

            prompt = f"""
            Analyze {symbol} for a {trade_type} trading strategy. The response must be in Turkish, at least 500 characters, and up to 5000 characters. Use the provided market data and technical indicators to create a unique, detailed, and context-aware analysis. Avoid repetitive or formulaic phrases, and ensure the tone is natural, professional, and engaging. Incorporate the group conversation history to personalize the analysis and directly address any questions or comments (e.g., 'Will it rise?').

            Market Data:
            - Current Price: {data['price']} USDT
            - Volume Change:
              - 60m: {data.get('indicators', {}).get('volume_change_60m', 'Unknown')}%
            - RSI:
              - 60m: {data.get('indicators', {}).get('rsi_60m', 'Unknown')}
            - MACD:
              - 60m: {data.get('indicators', {}).get('macd_60m', 'Unknown')}
            - Bid/Ask Ratio: {data.get('indicators', {}).get('bid_ask_ratio', 'Unknown')}

            Group Conversation History:
            {context_str if context_str else 'No group conversation history available.'}

            Include the following in your analysis, but the format and style are entirely up to you:
            - Recommended entry, exit, and stop-loss prices (in USDT, determined solely by you).
            - Leverage suggestion (e.g., 1x for spot, appropriate level for futures).
            - Pump and dump probabilities (as percentages, based on indicators).
            - Trend prediction (bullish, bearish, or neutral).
            - Support and resistance levels (in USDT).
            - Risk/reward ratio.
            - Fundamental analysis including market sentiment, volume trends, and buy/sell pressure.
            - A unique comment with a Buy/Sell/Hold recommendation, supported by detailed reasoning.

            Respond to any questions or comments in the group conversation history. Ensure the response is creative, avoids clichés, and provides actionable insights. Minimum 500 characters.
            """
            response = self.client.chat.completions.create(
                model="deepseek-moe",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000
            )
            analysis_text = response.choices[0].message.content
            logger.info(f"DeepSeek raw response for {symbol}: {analysis_text}")
            if len(analysis_text) < 500:
                logger.warning(f"DeepSeek response for {symbol} is less than 500 characters: {len(analysis_text)}")
                analysis_text += " " * (500 - len(analysis_text))
            return {'short_term': self.parse_deepseek_response(analysis_text, data['price'])}
        except Exception as e:
            logger.error(f"Error during DeepSeek analysis for {symbol}: {e}")
            return {
                'short_term': {
                    'entry_price': data['price'],
                    'exit_price': data['price'] * 1.02,
                    'stop_loss': data['price'] * 0.98,
                    'leverage': '1x' if trade_type == 'spot' else '3x',
                    'pump_probability': 50,
                    'dump_probability': 50,
                    'trend': 'Neutral',
                    'support_level': data['price'] * 0.95,
                    'resistance_level': data['price'] * 1.05,
                    'risk_reward_ratio': 1.0,
                    'fundamental_analysis': 'Analysis failed: Insufficient data.',
                    'comment': self.generate_dynamic_default_comment(data)
                }
            }

    def parse_deepseek_response(self, text, current_price):
        """Parses DeepSeek response robustly."""
        try:
            result = {
                'entry_price': None,
                'exit_price': None,
                'stop_loss': None,
                'leverage': '1x',
                'pump_probability': 50,
                'dump_probability': 50,
                'trend': 'Neutral',
                'support_level': None,
                'resistance_level': None,
                'risk_reward_ratio': 1.0,
                'fundamental_analysis': text[:500] if text else 'Analysis failed',
                'comment': text if text else 'Analysis failed: Insufficient data.'
            }

            lines = text.split('\n')
            for line in lines:
                line = line.strip().lower()
                number_match = re.search(r'\d+\.?\d*', line)
                if 'giriş fiyatı' in line and number_match:
                    result['entry_price'] = float(number_match.group(0))
                elif 'çıkış fiyatı' in line and number_match:
                    result['exit_price'] = float(number_match.group(0))
                elif 'stop loss' in line and number_match:
                    result['stop_loss'] = float(number_match.group(0))
                elif 'kaldıraç' in line:
                    result['leverage'] = line.split(':')[1].strip() if ':' in line else '1x'
                elif 'pump olasılığı' in line and re.search(r'\d+', line):
                    result['pump_probability'] = int(re.search(r'\d+', line).group(0))
                elif 'dump olasılığı' in line and re.search(r'\d+', line):
                    result['dump_probability'] = int(re.search(r'\d+', line).group(0))
                elif 'trend' in line:
                    result['trend'] = line.split(':')[1].strip() if ':' in line else 'Neutral'
                elif 'destek seviyesi' in line and number_match:
                    result['support_level'] = float(number_match.group(0))
                elif 'direnç seviyesi' in line and number_match:
                    result['resistance_level'] = float(number_match.group(0))
                elif 'risk/ödül oranı' in line and number_match:
                    result['risk_reward_ratio'] = float(number_match.group(0))
                elif 'temel analiz' in line:
                    result['fundamental_analysis'] = line.split(':')[1].strip()[:500] if ':' in line else text[:500]
                elif 'yorum' in line:
                    result['comment'] = line.split(':')[1].strip() if ':' in line else text

            # Fallback to current price only if DeepSeek fails to provide values
            if result['entry_price'] is None:
                result['entry_price'] = current_price
            if result['exit_price'] is None:
                result['exit_price'] = current_price * 1.02
            if result['stop_loss'] is None:
                result['stop_loss'] = current_price * 0.98
            if result['support_level'] is None:
                result['support_level'] = current_price * 0.95
            if result['resistance_level'] is None:
                result['resistance_level'] = current_price * 1.05

            return result
        except Exception as e:
            logger.error(f"Error parsing DeepSeek response: {e}")
            return {
                'entry_price': current_price,
                'exit_price': current_price * 1.02,
                'stop_loss': current_price * 0.98,
                'leverage': '1x',
                'pump_probability': 50,
                'dump_probability': 50,
                'trend': 'Neutral',
                'support_level': current_price * 0.95,
                'resistance_level': current_price * 1.05,
                'risk_reward_ratio': 1.0,
                'fundamental_analysis': 'Analysis failed',
                'comment': self.generate_dynamic_default_comment({'price': current_price})
            }

    def generate_dynamic_default_comment(self, data):
        """Generates a dynamic default comment based on indicators."""
        indicators = data.get('indicators', {})
        rsi_60m = indicators.get('rsi_60m', 0.0)
        volume_change_60m = indicators.get('volume_change_60m', 0.0)
        macd_60m = indicators.get('macd_60m', 0.0)
        bid_ask_ratio = indicators.get('bid_ask_ratio', 0.0)

        comment = "Market analysis indicates uncertainty at this time. "
        if rsi_60m > 70:
            comment += f"RSI at {rsi_60m:.2f} suggests overbought conditions, indicating potential for a price correction. "
        elif rsi_60m < 30:
            comment += f"RSI at {rsi_60m:.2f} indicates oversold conditions, possibly presenting a buying opportunity. "
        else:
            comment += f"RSI at {rsi_60m:.2f} is neutral, showing no clear trend direction. "

        if volume_change_60m > 100:
            comment += f"Volume surged by {volume_change_60m:.2f}%, suggesting increased market activity. "
        elif volume_change_60m < -50:
            comment += f"Volume dropped by {volume_change_60m:.2f}%, indicating a quiet market. "
        else:
            comment += f"Volume changed by {volume_change_60m:.2f}%, requiring close monitoring. "

        if macd_60m > 0:
            comment += f"MACD at {macd_60m:.2f} signals a bullish trend. "
        elif macd_60m < 0:
            comment += f"MACD at {macd_60m:.2f} suggests a bearish trend. "

        if bid_ask_ratio > 1.5:
            comment += f"Bid/ask ratio of {bid_ask_ratio:.2f} indicates strong buying pressure."
        elif bid_ask_ratio < 0.7:
            comment += f"Bid/ask ratio of {bid_ask_ratio:.2f} shows dominant selling pressure."
        else:
            comment += f"Bid/ask ratio of {bid_ask_ratio:.2f} reflects balanced buying and selling."

        if len(comment) < 500:
            comment += " " * (500 - len(comment))
        return comment[:500]

class Storage:
    """Stores analysis results and group conversations."""
    def __init__(self):
        self.file_path = "analysis.json"
        self.conversation_file_path = "conversations.json"

    def save_analysis(self, data):
        """Saves analysis results to a JSON file."""
        logger.info("Saving analysis results")
        try:
            existing_data = self.load_analysis()
            existing_data.update(data)
            with open(self.file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    def load_analysis(self):
        """Loads analysis results from a JSON file."""
        logger.info("Loading analysis results")
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return {}

    def save_conversation(self, chat_id, message, timestamp):
        """Saves group conversations."""
        logger.info(f"Saving conversation for chat_id={chat_id}")
        try:
            conversations = self.load_conversations()
            if str(chat_id) not in conversations:
                conversations[str(chat_id)] = []
            conversations[str(chat_id)].append({
                'message': message,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
            conversations[str(chat_id)] = conversations[str(chat_id)][-50:]
            with open(self.conversation_file_path, 'w') as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    def load_conversations(self):
        """Loads group conversations."""
        logger.info("Loading conversations")
        try:
            with open(self.conversation_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            return {}

def calculate_indicators(kline_60m, order_book):
    """Calculates technical indicators for 60m timeframe."""
    logger.info("Calculating technical indicators")
    try:
        indicators = {}
        if kline_60m and len(kline_60m) > 1:
            df = pd.DataFrame(
                kline_60m,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_timestamp', 'quote_volume']
            )
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            volume_change = (df['volume'].iloc[-1] / df['volume'].iloc[-2] - 1) * 100 if df['volume'].iloc[-2] != 0 else 0.0
            indicators['volume_change_60m'] = volume_change
            rsi = ta.rsi(df['close'], length=14)
            indicators['rsi_60m'] = rsi.iloc[-1] if not rsi.empty else 0.0
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            indicators['macd_60m'] = macd['MACD_12_26_9'].iloc[-1] if not macd.empty else 0.0
        else:
            indicators['volume_change_60m'] = 0.0
            indicators['rsi_60m'] = 0.0
            indicators['macd_60m'] = 0.0

        if order_book and 'bids' in order_book and 'asks' in order_book:
            bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
            ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
            indicators['bid_ask_ratio'] = bid_volume / ask_volume if ask_volume > 0 else 0.0
        else:
            indicators['bid_ask_ratio'] = 0.0

        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {
            'volume_change_60m': 0.0,
            'rsi_60m': 0.0,
            'macd_60m': 0.0,
            'bid_ask_ratio': 0.0
        }

def explain_indicators(indicators):
    """Explains indicators in a user-friendly way."""
    explanations = []
    volume_change = indicators.get('volume_change_60m', 0.0)
    rsi = indicators.get('rsi_60m', 0.0)
    macd = indicators.get('macd_60m', 0.0)
    
    if isinstance(volume_change, (int, float)):
        if volume_change > 100:
            vol_explain = f"60m: Volume increased by {volume_change:.2f}%, indicating strong market activity."
        elif volume_change > 0:
            vol_explain = f"60m: Volume rose by {volume_change:.2f}%, showing growing interest."
        elif volume_change < -50:
            vol_explain = f"60m: Volume dropped by {volume_change:.2f}%, suggesting a quiet market."
        else:
            vol_explain = f"60m: Volume changed by {volume_change:.2f}%, indicating stable activity."
    else:
        vol_explain = "60m: No volume data available."
    explanations.append(vol_explain)

    if isinstance(rsi, (int, float)):
        if rsi > 70:
            rsi_explain = f"60m: RSI at {rsi:.2f}, in overbought territory, suggesting a potential pullback."
        elif rsi < 30:
            rsi_explain = f"60m: RSI at {rsi:.2f}, in oversold territory, indicating a possible buying opportunity."
        else:
            rsi_explain = f"60m: RSI at {rsi:.2f}, in neutral territory with no clear signal."
    else:
        rsi_explain = "60m: No RSI data available."
    explanations.append(rsi_explain)

    if isinstance(macd, (int, float)):
        if macd > 0:
            macd_explain = f"60m: MACD at {macd:.2f}, signaling a bullish trend."
        elif macd < 0:
            macd_explain = f"60m: MACD at {macd:.2f}, indicating a bearish trend."
        else:
            macd_explain = f"60m: MACD at {macd:.2f}, showing a neutral trend."
    else:
        macd_explain = "60m: No MACD data available."
    explanations.append(macd_explain)

    bid_ask_ratio = indicators.get('bid_ask_ratio', 0.0)
    if isinstance(bid_ask_ratio, (int, float)):
        if bid_ask_ratio > 1.5:
            bid_ask_explain = f"Bid/Ask Ratio: {bid_ask_ratio:.2f}, indicating strong buying pressure."
        elif bid_ask_ratio < 0.7:
            bid_ask_explain = f"Bid/Ask Ratio: {bid_ask_ratio:.2f}, showing dominant selling pressure."
        else:
            bid_ask_explain = f"Bid/Ask Ratio: {bid_ask_ratio:.2f}, reflecting balanced buying and selling."
    else:
        bid_ask_explain = "Bid/Ask Ratio: No data available."
    explanations.append(bid_ask_explain)

    return "\n".join(explanations)

class TelegramBot:
    def __init__(self):
        """Initializes the Telegram bot."""
        logger.info("Initializing TelegramBot")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.storage = Storage()
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            logger.error("TELEGRAM_BOT_TOKEN is missing")
            raise ValueError("TELEGRAM_BOT_TOKEN is missing")
        try:
            self.app = Application.builder().token(bot_token).build()
            logger.info("Application initialized")
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("analyze", self.analyze))
            self.app.add_handler(CallbackQueryHandler(self.button))
            self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
            self.web_app = None
            self.active_analyses = {}
            if self.app.job_queue:
                self.app.job_queue.start()
                logger.info("Job queue started")
        except Exception as e:
            logger.error(f"Error initializing Application: {e}")
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Displays menu for the start command."""
        logger.info(f"Start command received, chat_id={update.effective_chat.id}")
        keyboard = [
            [InlineKeyboardButton("Top 10 Spot Analysis", callback_data='top_10_spot')],
            [InlineKeyboardButton("Top 100 Spot Analysis", callback_data='top_100_spot')],
            [InlineKeyboardButton("Top 10 Futures Analysis", callback_data='top_10_futures')],
            [InlineKeyboardButton("Top 100 Futures Analysis", callback_data='top_100_futures')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Use the buttons for analysis or the /analyze <symbol> [trade_type] command (e.g., /analyze BTCUSDT spot).",
            reply_markup=reply_markup
        )

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performs analysis for the specified symbol."""
        logger.info(f"Analyze command received: {update.message.text}")
        try:
            args = update.message.text.split()
            if len(args) < 2:
                await update.message.reply_text("Please provide a symbol. Example: /analyze BTCUSDT spot")
                return
            symbol = args[1].upper()
            trade_type = args[2].lower() if len(args) > 2 else 'spot'
            if trade_type not in ['spot', 'futures']:
                await update.message.reply_text("Trade type must be 'spot' or 'futures'. Example: /analyze BTCUSDT spot")
                return
            if not symbol.endswith('USDT'):
                await update.message.reply_text("Symbol must be a USDT pair. Example: /analyze BTCUSDT")
                return

            analysis_key = f"{symbol}_{trade_type}_{update.effective_chat.id}"
            if analysis_key in self.active_analyses:
                await update.message.reply_text(f"Analysis for {symbol} ({trade_type}) is already in progress. Please wait.")
                return
            self.active_analyses[analysis_key] = True

            mexc = MEXCClient()
            if not await mexc.validate_symbol(symbol):
                del self.active_analyses[analysis_key]
                await update.message.reply_text(f"Error: {symbol} is an invalid trading pair.")
                return

            await update.message.reply_text(f"Analyzing {symbol} for {trade_type}...")
            data = await self.process_coin(symbol, mexc, trade_type, update.effective_chat.id)
            await mexc.close()
            del self.active_analyses[analysis_key]

            if not data:
                await update.message.reply_text(f"Failed to analyze {symbol}.")
        except Exception as e:
            logger.error(f"Error in analyze command: {e}")
            await update.message.reply_text(f"Error: {str(e)}")
            if analysis_key in self.active_analyses:
                del self.active_analyses[analysis_key]

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles button clicks."""
        query = update.callback_query
        await query.answer()
        logger.info(f"Button clicked: {query.data}")
        try:
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            await query.message.reply_text(f"Performing {trade_type.upper()} analysis (Top {limit})...")
            data = {'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type}
            if self.app.job_queue is None:
                await self.analyze_and_send(context, data)
            else:
                self.app.job_queue.run_once(
                    self.analyze_and_send,
                    0,
                    data=data,
                    chat_id=self.group_id
                )
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.message.reply_text(f"Error: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE, data=None):
        """Performs analysis for top coins and sends results."""
        if data is None:
            data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.info(f"Analyzing: chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            if limit == 100 and trade_type == 'spot':
                results = await self.analyze_top_100_from_endpoints(chat_id, trade_type)
            else:
                results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"No results for Top {limit} {trade_type} analysis.")
            logger.info(f"Completed analysis for Top {limit} {trade_type}")
        except Exception as e:
            logger.error(f"Error in analyze_and_send: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"Error during {trade_type} analysis: {str(e)}")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Responds to non-command messages in the group."""
        message_text = update.message.text
        logger.info(f"Received message: {message_text}")
        try:
            self.storage.save_conversation(
                chat_id=update.effective_chat.id,
                message=message_text,
                timestamp=datetime.now()
            )
            deepseek = DeepSeekClient()
            conversations = self.storage.load_conversations()
            group_context = conversations.get(str(update.effective_chat.id), [])
            context_str = "\n".join([f"[{c['timestamp']}] {c['message']}" for c in group_context])

            prompt = f"""
            Respond to the following message in Turkish, providing a detailed and context-aware answer based on the group conversation history. The response should be at least 500 characters, natural, professional, and engaging. If the message is a question about a specific coin (e.g., 'Will BTCUSDT rise?'), provide a brief market analysis based on general knowledge or recent trends, and address the question directly. Avoid repetitive phrases and incorporate the conversation history for personalization.

            Message: {message_text}
            Group Conversation History:
            {context_str if context_str else 'No group conversation history available.'}
            """
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    deepseek.client.chat.completions.create,
                    model="deepseek-moe",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5000
                ),
                timeout=20
            )
            response_text = response.choices[0].message.content
            if len(response_text) < 500:
                response_text += " " * (500 - len(response_text))
            await update.message.reply_text(response_text)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(f"Error: {str(e)}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Displays saved analysis results."""
        logger.info("Show_analysis command received")
        try:
            data = self.storage.load_analysis()
            messages = []
            for key, coin_data in data.items():
                if isinstance(coin_data, dict) and 'coin' in coin_data:
                    trade_type = key.split('_')[-1]
                    message = self.format_results(coin_data, trade_type, coin_data['coin'])
                    messages.append(message)
            if messages:
                await update.message.reply_text("\n\n".join(messages))
            else:
                await update.message.reply_text("No analysis results found.")
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            await update.message.reply_text(f"Error: {str(e)}")

    def format_results(self, coin_data, trade_type, symbol):
        """Formats analysis results."""
        logger.info(f"Formatting results for {symbol} ({trade_type})")
        indicators = coin_data.get('indicators', {})
        analysis = coin_data.get('deepseek_analysis', {}).get('short_term', {})
        volume_changes = indicators.get('volume_change_60m', 'Unknown')
        bid_ask_ratio = indicators.get('bid_ask_ratio', 'Unknown')
        rsi_values = indicators.get('rsi_60m', 'Unknown')
        macd_values = indicators.get('macd_60m', 'Unknown')
        try:
            message = (
                f" {symbol} {trade_type.upper()} Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                f"- Short-Term:\n"
                f"  - Entry: ${analysis.get('entry_price', 'Not specified')}\n"
                f"  - Exit: ${analysis.get('exit_price', 'Not specified')}\n"
                f"  - Stop Loss: ${analysis.get('stop_loss', 'Not specified')}\n"
                f"  - Leverage: {analysis.get('leverage', 'Unknown')}\n"
                f"- Trend: {analysis.get('trend', 'Neutral')}\n"
                f"- Pump Probability: {analysis.get('pump_probability', 0)}%\n"
                f"- Dump Probability: {analysis.get('dump_probability', 0)}%\n"
                f"- Support Level: ${analysis.get('support_level', 'Not specified')}\n"
                f"- Resistance Level: ${analysis.get('resistance_level', 'Not specified')}\n"
                f"- Risk/Reward Ratio: {analysis.get('risk_reward_ratio', 0):.2f}\n"
                f"- Fundamental Analysis: {analysis.get('fundamental_analysis', 'No data')}\n"
                f"- Indicators:\n"
                f"  - Volume Change: 60m: {volume_changes if isinstance(volume_changes, (int, float)) else 'Unknown'}%\n"
                f"  - Bid/Ask Ratio: {bid_ask_ratio if isinstance(bid_ask_ratio, (int, float)) else 'Unknown'}\n"
                f"  - RSI: 60m: {rsi_values if isinstance(rsi_values, (int, float)) else 'Unknown'}\n"
                f"  - MACD: 60m: {macd_values if isinstance(macd_values, (int, float)) else 'Unknown'}\n"
                f"- Indicator Explanations:\n{explain_indicators(indicators)}\n"
                f"- DeepSeek Comment: {analysis.get('comment', 'No comment available.')}"
            )
            return message
        except Exception as e:
            logger.error(f"Error formatting results for {symbol}: {e}")
            return f"Error formatting {symbol} analysis: {str(e)}"

    async def process_coin(self, symbol, mexc, trade_type, chat_id, endpoint=None):
        """Processes a single coin for analysis."""
        logger.info(f"Processing {symbol} ({trade_type})")
        try:
            data = await mexc.fetch_and_save_market_data(symbol, endpoint)
            if not data or not data.get('klines', {}).get('60m'):
                logger.warning(f"No valid market data for {symbol} ({trade_type})")
                await self.app.bot.send_message(chat_id=chat_id, text=f"No valid market data for {symbol}")
                return None

            data['indicators'] = calculate_indicators(
                data['klines'].get('60m', []),
                data.get('order_book')
            )
            if not data['indicators']:
                logger.warning(f"Failed to calculate indicators for {symbol} ({trade_type})")
                await self.app.bot.send_message(chat_id=chat_id, text=f"Failed to calculate indicators for {symbol}")
                return None

            deepseek = DeepSeekClient()
            data['deepseek_analysis'] = deepseek.analyze_coin(symbol, data, trade_type, chat_id)
            data['coin'] = symbol

            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Analysis sent for {symbol} ({trade_type})")

            self.storage.save_analysis({f'{symbol}_{trade_type}': data})
            return data
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"Error processing {symbol}: {str(e)}")
            return None

    async def analyze_coins(self, limit, trade_type, chat_id):
        """Analyzes top coins."""
        logger.info(f"Starting analyze_coins: limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_{limit}_{trade_type}': []}
        coins = await mexc.get_top_coins(limit)
        logger.info(f"Analyzing {len(coins)} coins: {coins[:5]}...")

        for symbol in coins:
            analysis_key = f"{symbol}_{trade_type}_{chat_id}"
            if analysis_key in self.active_analyses:
                logger.info(f"Analysis for {symbol} already in progress, skipping")
                continue
            self.active_analyses[analysis_key] = True
            coin_data = await self.process_coin(symbol, mexc, trade_type, chat_id)
            if coin_data:
                results[f'top_{limit}_{trade_type}'].append(coin_data)
            del self.active_analyses[analysis_key]
            await asyncio.sleep(1)
        await mexc.close()
        return results

    async def analyze_top_100_from_endpoints(self, chat_id, trade_type):
        """Analyzes top 100 coins from endpoints.json."""
        logger.info(f"Analyzing top 100 {trade_type} from endpoints.json")
        mexc = MEXCClient()
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_100_{trade_type}': []}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ENDPOINTS_JSON_URL) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch endpoints.json: {response.status}")
                        await self.app.bot.send_message(chat_id=chat_id, text="Failed to fetch endpoints.json.")
                        return results
                    endpoints = await response.json()
            
            for entry in endpoints:
                symbol = entry['symbol']
                endpoint = entry['endpoint']
                analysis_key = f"{symbol}_{trade_type}_{chat_id}"
                if analysis_key in self.active_analyses:
                    logger.info(f"Analysis for {symbol} already in progress, skipping")
                    continue
                self.active_analyses[analysis_key] = True
                coin_data = await self.process_coin(symbol, mexc, trade_type, chat_id, endpoint)
                if coin_data:
                    results[f'top_100_{trade_type}'].append(coin_data)
                del self.active_analyses[analysis_key]
                await asyncio.sleep(1)
            await mexc.close()
            return results
        except Exception as e:
            logger.error(f"Error during top 100 analysis: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"Error during top 100 analysis: {str(e)}")
            return results

    async def webhook_handler(self, request):
        """Handles webhook requests."""
        logger.info("Webhook request received")
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
        """Starts the webhook server."""
        logger.info("Starting webhook server")
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        try:
            await self.app.bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to {webhook_url}")
        except Exception as e:
            logger.error(f"Error setting webhook: {e}")
            raise
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook server running on port {os.getenv('PORT', 8443)}")
        await asyncio.Event().wait()

def main():
    logger.info("Starting main")
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    logger.info("Script started")
    main()

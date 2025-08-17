import os
import json
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
from dotenv import load_dotenv
from openai import OpenAI
from aiohttp import web
import logging
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TelegramBot:
    def __init__(self):
        logger.debug("Initializing TelegramBot")
        self.group_id = os.getenv('TELEGRAM_GROUP_ID')
        if not self.group_id:
            logger.error("TELEGRAM_GROUP_ID is not set in environment variables")
            raise ValueError("TELEGRAM_GROUP_ID is not set")
        try:
            self.group_id = int(self.group_id)
            logger.debug(f"Group ID set to: {self.group_id}")
        except ValueError:
            logger.error(f"Invalid TELEGRAM_GROUP_ID format: {self.group_id}")
            raise ValueError(f"Invalid TELEGRAM_GROUP_ID format: {self.group_id}")
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        logger.debug("OpenAI client initialized for DeepSeek")
        self.app = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        self.web_app = None
        logger.debug("Telegram Application initialized")

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
        logger.debug("Handlers added to Telegram Application")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug(f"Start command received from chat_id={update.effective_chat.id}")
        keyboard = [
            [InlineKeyboardButton("Top 100 Spot Analizi", callback_data='top_100_spot')],
            [InlineKeyboardButton("Top 300 Spot Analizi", callback_data='top_300_spot')],
            [InlineKeyboardButton("Top 100 Vadeli Analizi", callback_data='top_100_futures')],
            [InlineKeyboardButton("Top 300 Vadeli Analizi", callback_data='top_300_futures')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Analiz için butonları kullanabilirsiniz:", reply_markup=reply_markup)
        logger.info("Start command processed")

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        logger.debug(f"Button clicked: {query.data} from chat_id={query.message.chat_id}")
        await query.answer()
        try:
            if not context.job_queue:
                logger.error("JobQueue is not available. Ensure python-telegram-bot[job-queue] is installed.")
                await query.message.reply_text("Error: JobQueue is not configured. Contact the bot administrator.")
                return
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            logger.debug(f"Starting analysis for limit={limit}, trade_type={trade_type}")
            await query.message.reply_text(f"{trade_type.upper()} analizi yapılıyor (Top {limit})...")
            context.job_queue.run_once(
                self.analyze_and_send,
                0,
                data={'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type},
                chat_id=self.group_id
            )
            logger.info(f"Scheduled analysis job for Top {limit} {trade_type}")
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.message.reply_text(f"Error during analysis initiation: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE):
        data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.debug(f"Analyzing and sending results for chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            results = await self.analyze_coins(limit, trade_type)
            logger.debug(f"Analysis results: {results}")
            message = self.format_results(results, trade_type)
            logger.debug(f"Formatted message: {message[:200]}...")
            if message.strip().endswith(f"TOP_{limit}_{trade_type.upper()}:\n"):
                logger.warning(f"No analysis results for Top {limit} {trade_type}")
                message = f"No significant results for Top {limit} {trade_type} analysis. Please try again later."
            await context.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Analysis sent to Telegram for Top {limit} {trade_type}: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error in analyze_and_send: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Error during {trade_type} analysis: {str(e)}"
            )

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug(f"Chat message received: {update.message.text} from chat_id={update.effective_chat.id}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": update.message.text}]
                ),
                timeout=20
            )
            reply_text = response.choices[0].message.content
            logger.debug(f"DeepSeek response: {reply_text[:100]}...")
            await update.message.reply_text(reply_text)
            logger.info(f"DeepSeek response sent for message: {update.message.text[:50]}...")
        except asyncio.TimeoutError:
            logger.error("DeepSeek API timed out")
            await update.message.reply_text("Error: DeepSeek API timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(f"Error in chat: {str(e)}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.debug("Show analysis command received")
        try:
            with open('analysis.json', 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded analysis data: {data}")
            message_spot = self.format_results(data, 'spot')
            message_futures = self.format_results(data, 'futures')
            logger.debug(f"Formatted spot message: {message_spot[:200]}...")
            logger.debug(f"Formatted futures message: {message_futures[:200]}...")
            messages = []
            if not message_spot.strip().endswith("TOP_100_SPOT:\n") and not message_spot.strip().endswith("TOP_300_SPOT:\n"):
                messages.append(message_spot)
            if not message_futures.strip().endswith("TOP_100_FUTURES:\n") and not message_futures.strip().endswith("TOP_300_FUTURES:\n"):
                messages.append(message_futures)
            if messages:
                await update.message.reply_text("\n\n".join(messages))
                logger.info("Analysis messages sent to Telegram")
            else:
                await update.message.reply_text("No significant analysis results found in the stored data.")
                logger.info("No significant analysis results found")
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            await update.message.reply_text("No analysis found.")
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            await update.message.reply_text(f"Error loading analysis: {str(e)}")

    def format_results(self, data, trade_type):
        logger.debug(f"Formatting results for trade_type={trade_type}")
        message = f"📊 {trade_type.upper()} Günlük Coin Analizi ({data.get('date', 'Unknown')})\n"
        for category in [f'top_100_{trade_type}', f'top_300_{trade_type}']:
            logger.debug(f"Processing category: {category}")
            message += f"\n{category.upper()}:\n"
            coins = data.get(category, [])
            if not coins:
                message += "No coins analyzed yet.\n"
                logger.debug(f"No coins for {category}")
                continue
            coins.sort(
                key=lambda x: x.get('deepseek_analysis', {}).get('short_term', {}).get('pump_probability', 0),
                reverse=True
            )
            for i, coin in enumerate(coins[:10], 1):
                logger.debug(f"Formatting coin {i}: {coin.get('coin', 'Unknown')}")
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
                    f"- Hacim Değişimleri: 1h: {indicators.get('volume_change_1h', 'N/A'):.2f}% | "
                    f"3h: {indicators.get('volume_change_3h', 'N/A'):.2f}% | "
                    f"6h: {indicators.get('volume_change_6h', 'N/A'):.2f}% | "
                    f"24h: {indicators.get('volume_change_24h', 'N/A'):.2f}%\n"
                    f"- Bid/Ask Oranı: {indicators.get('bid_ask_ratio', 'N/A'):.2f}\n"
                )
        logger.debug(f"Formatted message for {trade_type}: {message[:200]}...")
        return message

    async def process_coin(self, symbol, mexc, deepseek, trade_type):
        logger.debug(f"Processing coin: {symbol} ({trade_type})")
        try:
            data = await mexc.get_market_data(symbol)
            logger.debug(f"Market data for {symbol}: {data}")
            if not data:
                logger.warning(f"No market data for {symbol} ({trade_type})")
                return None
            from indicators import calculate_indicators
            logger.debug(f"Calling calculate_indicators for {symbol}")
            data['indicators'] = calculate_indicators(data['klines']['60m'], data['klines']['4h'], data.get('order_book'))
            logger.debug(f"Indicators for {symbol}: {data['indicators']}")
            if not data['indicators']:
                logger.warning(f"No indicators calculated for {symbol} ({trade_type})")
                return None
            logger.debug(f"Calling deepseek.analyze_coin for {symbol}")
            data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
            logger.debug(f"DeepSeek analysis for {symbol}: {data['deepseek_analysis']}")
            logger.info(f"Processed {symbol} ({trade_type}) successfully: price={data.get('price', 'N/A')}, indicators={list(data['indicators'].keys())[:5]}...")
            return data
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            return None

    async def analyze_coins(self, limit, trade_type):
        from mexc_api import MEXCClient
        from deepseek import DeepSeekClient
        from storage import Storage

        logger.debug(f"Starting analyze_coins for limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        deepseek = DeepSeekClient()
        storage = Storage()
        
        logger.debug("Fetching top coins")
        coins = await mexc.get_top_coins(limit)
        logger.info(f"Starting analysis for {len(coins)} coins ({trade_type}): {coins[:5]}...")
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_100_{trade_type}': [], f'top_300_{trade_type}': []}
        
        logger.debug(f"Creating tasks for {len(coins)} coins")
        tasks = [self.process_coin(symbol, mexc, deepseek, trade_type) for symbol in coins]
        coin_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_coins = []
        for symbol, data in zip(coins, coin_data):
            logger.debug(f"Processing result for {symbol}: {data}")
            if data and not isinstance(data, Exception):
                valid_coins.append(data)
            elif isinstance(data, Exception):
                logger.error(f"Exception in processing {symbol}: {data}")
            else:
                logger.warning(f"No data returned for {symbol}")
        
        results[f'top_100_{trade_type}' if limit == 100 else f'top_300_{trade_type}'] = valid_coins
        logger.info(f"Processed {len(valid_coins)} valid coins for Top {limit} {trade_type}")
        logger.debug(f"Saving analysis results: {results}")
        storage.save_analysis(results)
        logger.debug("Closing MEXC client")
        await mexc.close()
        logger.debug(f"Returning analysis results: {len(valid_coins)} valid coins")
        return results

    async def webhook_handler(self, request):
        logger.debug("Webhook request received")
        try:
            raw_data = await request.json()
            logger.debug(f"Raw webhook data: {raw_data}")
            update = Update.de_json(raw_data, self.app.bot)
            if not update:
                logger.warning("Failed to parse webhook update")
                return web.Response(text="ERROR: Invalid update", status=400)
            logger.info(f"Processing update: message={getattr(update.message, 'text', 'N/A')}, chat_id={getattr(update.effective_chat, 'id', 'N/A')}")
            await self.app.process_update(update)
            logger.debug("Webhook update processed")
            return web.Response(text="OK")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding webhook JSON: {e}")
            return web.Response(text="ERROR: Invalid JSON", status=400)
        except Exception as e:
            logger.error(f"Error in webhook handler: {e}")
            return web.Response(text=f"ERROR: {str(e)}", status=500)

    async def run(self):
        logger.debug("Starting webhook server")
        await self.app.initialize()
        logger.debug("Telegram Application initialized")
        await self.app.start()
        logger.debug("Telegram Application started")
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        logger.debug(f"Setting webhook to {webhook_url}")
        await self.app.bot.set_webhook(url=webhook_url)
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook server running on port {os.getenv('PORT', 8443)}")
        await asyncio.Event().wait()

def main():
    logger.debug("Starting main function")
    bot = TelegramBot()
    asyncio.run(bot.run())
    logger.debug("Main function completed")

if __name__ == "__main__":
    logger.debug("Script started")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_coins":
        limit = 100 if len(sys.argv) < 3 else int(sys.argv[2])
        trade_type = sys.argv[3] if len(sys.argv) > 3 else 'spot'
        logger.debug(f"Running analyze_coins with limit={limit}, trade_type={trade_type}")
        bot = TelegramBot()
        asyncio.run(bot.analyze_coins(limit, trade_type))
    else:
        main()
    logger.debug("Script completed")

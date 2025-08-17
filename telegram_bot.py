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

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# MEXCClient, DeepSeekClient, Storage, calculate_indicators aynı kalıyor
# (Paylaştığınız kodun bu kısımları zaten doğru ve çalışır durumda)

class TelegramBot:
    def __init__(self):
        logger.debug("Initializing TelegramBot")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            logger.error("TELEGRAM_BOT_TOKEN is not set")
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is missing")
        
        try:
            self.app = Application.builder().token(bot_token).build()
            logger.debug("Application initialized successfully")
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CallbackQueryHandler(self.button))
            self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
            self.web_app = None
            # Job queue başlatma
            try:
                self.app.job_queue.start()
                logger.debug("Job queue started successfully")
            except Exception as e:
                logger.warning(f"Failed to start job_queue: {e}. Proceeding without job_queue.")
                self.app.job_queue = None  # Job queue başlatılamazsa None olarak bırak
        except Exception as e:
            logger.error(f"Error initializing Application: {e}")
            raise

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
            data = {'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type}
            if context.job_queue is None:
                logger.warning("context.job_queue is None, running analyze_and_send directly")
                await self.analyze_and_send(context, data)
            else:
                context.job_queue.run_once(
                    self.analyze_and_send,
                    0,
                    data=data,
                    chat_id=self.group_id
                )
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.message.reply_text(f"Hata: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE, data=None):
        if data is None:
            data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.debug(f"Analyzing for chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"Top {limit} {trade_type} analizi için anlamlı sonuç bulunamadı.")
            logger.info(f"Analysis completed for Top {limit} {trade_type}")
        except Exception as e:
            logger.error(f"Error in analyze_and_send: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"{trade_type} analizi sırasında hata: {str(e)}")

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
            messages = []
            for trade_type in ['spot', 'futures']:
                for limit in [100, 300]:
                    key = f'top_{limit}_{trade_type}'
                    if key in data and data[key]:
                        for coin_data in data[key]:
                            message = self.format_results(coin_data, trade_type, coin_data['coin'])
                            messages.append(message)
            if messages:
                await update.message.reply_text("\n\n".join(messages))
            else:
                await update.message.reply_text("Analiz sonucu bulunamadı.")
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            await update.message.reply_text(f"Hata: {str(e)}")

    def format_results(self, coin_data, trade_type, symbol):
        logger.debug(f"Formatting results for {symbol} ({trade_type})")
        indicators = coin_data.get('indicators', {})
        analysis = coin_data.get('deepseek_analysis', {}).get('short_term', {})
        # Hacim değişimlerini güvenli bir şekilde biçimlendir
        volume_changes = {}
        for tf in ['1m', '5m', '15m', '30m', '60m']:
            value = indicators.get(f'volume_change_{tf}', None)
            volume_changes[tf] = f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"
        bid_ask_ratio = indicators.get('bid_ask_ratio', None)
        bid_ask_ratio_str = f"{bid_ask_ratio:.2f}" if isinstance(bid_ask_ratio, (int, float)) else "N/A"
        try:
            message = (
                f"📊 {symbol} {trade_type.upper()} Analizi ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                f"- Kısa Vadeli: Giriş: ${analysis.get('entry_price', 0):.2f} | "
                f"Çıkış: ${analysis.get('exit_price', 0):.2f} | "
                f"Stop Loss: ${analysis.get('stop_loss', 0):.2f} | "
                f"Kaldıraç: {analysis.get('leverage', 'N/A')}\n"
                f"- Pump Olasılığı: {analysis.get('pump_probability', 0)}% | "
                f"Dump Olasılığı: {analysis.get('dump_probability', 0)}%\n"
                f"- Temel Analiz: {analysis.get('fundamental_analysis', 'Veri yok')}\n"
                f"- Hacim Değişimleri: 1m: {volume_changes['1m']}% | "
                f"5m: {volume_changes['5m']}% | "
                f"15m: {volume_changes['15m']}% | "
                f"30m: {volume_changes['30m']}% | "
                f"60m: {volume_changes['60m']}% \n"
                f"- Bid/Ask Oranı: {bid_ask_ratio_str}\n"
            )
            return message
        except Exception as e:
            logger.error(f"Error formatting results for {symbol}: {e}")
            return f"Error formatting {symbol} analysis: {str(e)}"

    async def process_coin(self, symbol, mexc, deepseek, trade_type, chat_id):
        logger.debug(f"Processing coin: {symbol} ({trade_type})")
        try:
            data = await mexc.fetch_and_save_market_data(symbol)
            if not data:
                logger.warning(f"No valid market data for {symbol} ({trade_type})")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için geçerli piyasa verisi yok")
                return None

            data['indicators'] = calculate_indicators(
                data['klines'].get('1m', []), data['klines'].get('5m', []), data['klines'].get('15m', []),
                data['klines'].get('30m', []), data['klines'].get('60m', []), data.get('order_book')
            )
            if not data['indicators']:
                logger.warning(f"No indicators for {symbol} ({trade_type})")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için gösterge hesaplanamadı")
                return None

            data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
            logger.info(f"Processed {symbol} ({trade_type}): price={data.get('price')}, "
                       f"klines_60m={len(data['klines'].get('60m', []))}")

            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Analysis sent for {symbol} ({trade_type})")

            storage = Storage()
            storage.save_analysis({f'{symbol}_{trade_type}': [data]})

            return data
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} işlenirken hata: {str(e)}")
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
            await asyncio.sleep(3.0)  # Rate limit için artırıldı
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
        try:
            await self.app.bot.set_webhook(url=webhook_url)
            logger.debug(f"Webhook set to {webhook_url}")
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

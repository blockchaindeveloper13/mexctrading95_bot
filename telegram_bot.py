import os
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from openai import OpenAI
from aiohttp import web
import asyncio
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TelegramBot:
    def __init__(self, analyze_callback):
        self.analyze_callback = analyze_callback
        self.group_id = os.getenv('TELEGRAM_GROUP_ID')
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.app = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        self.web_app = None

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("Start command received")
        keyboard = [
            [InlineKeyboardButton("Top 100 Analiz Yap", callback_data='top_100')],
            [InlineKeyboardButton("Top 300 Analiz Yap", callback_data='top_300')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Analiz için butonları kullanabilirsiniz:", reply_markup=reply_markup)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        logger.info(f"Button clicked: {query.data}")
        limit = 100 if query.data == 'top_100' else 300
        results = self.analyze_callback(limit)
        message = self.format_results(results)
        await context.bot.send_message(chat_id=self.group_id, text=message)

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"Chat message received: {update.message.text}")
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": update.message.text}]
            )
            await update.message.reply_text(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(f"Error in chat: {e}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("Show analysis command received")
        try:
            with open('analysis.json', 'r') as f:
                data = json.load(f)
            message = self.format_results(data)
            await update.message.reply_text(message)
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            await update.message.reply_text("No analysis found.")

    def format_results(self, data):
        message = f"📊 Günlük Coin Analizi ({data['date']})\n"
        for category in ['top_100', 'top_300']:
            message += f"\n{category.upper()}:\n"
            for coin in data[category]:
                if coin['deepseek_analysis']['short_term'].get('pump_probability', 0) >= 70 or \
                   coin['deepseek_analysis']['short_term'].get('dump_probability', 0) >= 70:
                    message += f"1. {coin['coin']}\n"
                    message += f"- Kısa Vadeli: Giriş: ${coin['deepseek_analysis']['short_term'].get('entry_price', 0)} | Çıkış: ${coin['deepseek_analysis']['short_term'].get('exit_price', 0)} | Stop Loss: ${coin['deepseek_analysis']['short_term'].get('stop_loss', 0)} | Kaldıraç: {coin['deepseek_analysis']['short_term'].get('leverage', 'N/A')}\n"
                    message += f"- DeepSeek: {json.dumps(coin['deepseek_analysis']['short_term'])}\n"
        return message

    async def webhook_handler(self, request):
        logger.info("Webhook request received")
        try:
            update = Update.de_json(await request.json(), self.app.bot)
            await self.app.process_update(update)
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Error in webhook handler: {e}")
            return web.Response(text="ERROR", status=500)

    async def run(self):
        logger.info("Starting webhook server")
        # Application'ı başlat
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        logger.info(f"Setting webhook to {webhook_url}")
        await self.app.bot.set_webhook(url=webhook_url)
        # Web server'ı başlat
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook server running on port {os.getenv('PORT', 8443)}")
        # Süresiz çalış
        await asyncio.Event().wait()

def main():
    from mexc_api import MEXCClient
    from indicators import calculate_indicators
    from deepseek import DeepSeekClient
    bot = TelegramBot(lambda limit: analyze_coins(limit))
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()

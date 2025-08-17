import os
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from openai import OpenAI
from aiohttp import web
import asyncio
import logging
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TelegramBot:
    def __init__(self):
        self.group_id = os.getenv('TELEGRAM_GROUP_ID')
        if not self.group_id:
            logger.error("TELEGRAM_GROUP_ID is not set in environment variables")
            raise ValueError("TELEGRAM_GROUP_ID is not set")
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
        results = await self.analyze_coins(limit)
        message = self.format_results(results)
        if self.group_id:
            await context.bot.send_message(chat_id=self.group_id, text=message)
        else:
            logger.error("Cannot send message: group_id is not set")
            await query.message.reply_text("Error: Group ID is not set")

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

    async def analyze_coins(self, limit):
        from mexc_api import MEXCClient
        from indicators import calculate_indicators
        from deepseek import DeepSeekClient
        from storage import Storage

        mexc = MEXCClient()
        deepseek = DeepSeekClient()
        storage = Storage()
        
        coins = await mexc.get_top_coins(limit)
        results = {'date': datetime.now().strftime('%Y-%m-%d'), 'top_100': [], 'top_300': []}
        
        for symbol in coins:
            data = await mexc.get_market_data(symbol)
            if data:
                data['indicators'] = calculate_indicators(data['klines_1h'], data['klines_4h'])
                data['deepseek_analysis'] = deepseek.analyze_coin(data)
                results['top_100' if limit == 100 else 'top_300'].append(data)
        
        storage.save_analysis(results)
        await mexc.close()  # MEXC client’ı kapat
        return results

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
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_coins":
        limit = 100 if len(sys.argv) < 3 else int(sys.argv[2])
        bot = TelegramBot()
        asyncio.run(bot.analyze_coins(limit))
    else:
        main()

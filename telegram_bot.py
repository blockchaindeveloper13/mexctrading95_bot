import os
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
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
        try:
            self.group_id = int(self.group_id)
        except ValueError:
            logger.error(f"Invalid TELEGRAM_GROUP_ID format: {self.group_id}")
            raise ValueError(f"Invalid TELEGRAM_GROUP_ID format: {self.group_id}")
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
        logger.info(f"Button clicked: {query.data}")
        try:
            if not context.job_queue:
                logger.error("JobQueue is not available. Ensure python-telegram-bot[job-queue] is installed.")
                await query.message.reply_text("Error: JobQueue is not configured. Contact the bot administrator.")
                return
            # Parse callback data
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            # Hemen kullanıcıya bilgi ver
            message = await query.message.reply_text(f"{trade_type.upper()} analizi yapılıyor (Top {limit})...")
            # Analizi background task olarak başlat
            context.job_queue.run_once(
                self.analyze_and_send,
                0,
                data={'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type, 'query_message_id': message.message_id},
                chat_id=self.group_id
            )
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            await query.message.reply_text(f"Error during analysis initiation: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE):
        data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        query_message_id = data['query_message_id']
        try:
            results = await self.analyze_coins(limit, trade_type)
            message = self.format_results(results, trade_type)
            if message.strip().endswith(f"TOP_{limit}_{trade_type.upper()}:\n"):
                logger.warning(f"No analysis results for Top {limit} {trade_type}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"No significant results for Top {limit} {trade_type} analysis.",
                    reply_to_message_id=query_message_id
                )
            else:
                await context.bot.send_message(chat_id=chat_id, text=message, reply_to_message_id=query_message_id)
        except Exception as e:
            logger.error(f"Error in analyze_and_send: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Error during {trade_type} analysis: {str(e)}",
                reply_to_message_id=query_message_id
            )

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
            message_spot = self.format_results(data, 'spot')
            message_futures = self.format_results(data, 'futures')
            messages = []
            if not message_spot.strip().endswith("TOP_100_SPOT:\n") or not message_spot.strip().endswith("TOP_300_SPOT:\n"):
                messages.append(message_spot)
            if not message_futures.strip().endswith("TOP_100_FUTURES:\n") or not message_futures.strip().endswith("TOP_300_FUTURES:\n"):
                messages.append(message_futures)
            if messages:
                await update.message.reply_text("\n\n".join(messages))
            else:
                await update.message.reply_text("No significant analysis results found in the stored data.")
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            await update.message.reply_text("No analysis found.")

    def format_results(self, data, trade_type):
        message = f"📊 {trade_type.upper()} Günlük Coin Analizi ({data['date']})\n"
        for category in [f'top_100_{trade_type}', f'top_300_{trade_type}']:
            message += f"\n{category.upper()}:\n"
            # Tüm coin’leri al, olasılık sıralı
            coins = data.get(category, [])
            # Pump olasılığına göre sırala
            coins.sort(
                key=lambda x: x.get('deepseek_analysis', {}).get('short_term', {}).get('pump_probability', 0),
                reverse=True
            )
            # En fazla 10 coin göster
            for i, coin in enumerate(coins[:10], 1):
                message += f"{i}. {coin.get('coin', 'Unknown')}\n"
                message += (
                    f"- Kısa Vadeli: Giriş: ${coin['deepseek_analysis']['short_term'].get('entry_price', 0):.2f} | "
                    f"Çıkış: ${coin['deepseek_analysis']['short_term'].get('exit_price', 0):.2f} | "
                    f"Stop Loss: ${coin['deepseek_analysis']['short_term'].get('stop_loss', 0):.2f} | "
                    f"Kaldıraç: {coin['deepseek_analysis']['short_term'].get('leverage', 'N/A')}\n"
                    f"- Pump Olasılığı: {coin['deepseek_analysis']['short_term'].get('pump_probability', 0)}% | "
                    f"Dump Olasılığı: {coin['deepseek_analysis']['short_term'].get('dump_probability', 0)}%\n"
                    f"- Temel Analiz: {coin['deepseek_analysis']['short_term'].get('fundamental_analysis', 'No data')}\n"
                )
        return message

    async def process_coin(self, symbol, mexc, deepseek, trade_type):
        try:
            data = await mexc.get_market_data(symbol)
            if data:
                from indicators import calculate_indicators
                data['indicators'] = calculate_indicators(data['klines_1h'], data['klines_4h'])
                if data['indicators']:
                    data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
                    logger.info(f"Processed {symbol} ({trade_type}) successfully")
                    return data
                else:
                    logger.warning(f"No indicators calculated for {symbol} ({trade_type})")
            else:
                logger.warning(f"No market data for {symbol} ({trade_type})")
            return None
        except Exception as e:
            logger.error(f"Error processing {symbol} ({trade_type}): {e}")
            return None

    async def analyze_coins(self, limit, trade_type):
        from mexc_api import MEXCClient
        from deepseek import DeepSeekClient
        from storage import Storage

        mexc = MEXCClient()
        deepseek = DeepSeekClient()
        storage = Storage()
        
        coins = await mexc.get_top_coins(limit)
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_100_{trade_type}': [], f'top_300_{trade_type}': []}
        
        tasks = [self.process_coin(symbol, mexc, deepseek, trade_type) for symbol in coins]
        coin_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        for data in coin_data:
            if data and not isinstance(data, Exception):
                results[f'top_100_{trade_type}' if limit == 100 else f'top_300_{trade_type}'].append(data)
        
        storage.save_analysis(results)
        await mexc.close()
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
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        logger.info(f"Setting webhook to {webhook_url}")
        await self.app.bot.set_webhook(url=webhook_url)
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook server running on port {os.getenv('PORT', 8443)}")
        await asyncio.Event().wait()

def main():
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_coins":
        limit = 100 if len(sys.argv) < 3 else int(sys.argv[2])
        trade_type = sys.argv[3] if len(sys.argv) > 3 else 'spot'
        bot = TelegramBot()
        asyncio.run(bot.analyze_coins(limit, trade_type))
    else:
        main()

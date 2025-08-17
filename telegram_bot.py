from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

class TelegramBot:
    def __init__(self, analyze_callback):
        self.app = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        self.analyze_callback = analyze_callback
        self.group_id = os.getenv('TELEGRAM_GROUP_ID')
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    async def start(self, update: Update, context):
        keyboard = [
            [InlineKeyboardButton("Top 100 Analiz Yap", callback_data='top_100')],
            [InlineKeyboardButton("Top 300 Analiz Yap", callback_data='top_300')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Analiz için butonları kullanabilirsiniz:", reply_markup=reply_markup)

    async def button(self, update: Update, context):
        query = update.callback_query
        await query.answer()
        limit = 100 if query.data == 'top_100' else 300
        results = self.analyze_callback(limit)
        message = self.format_results(results)
        await context.bot.send_message(chat_id=self.group_id, text=message)

    async def chat(self, update: Update, context):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": update.message.text}]
            )
            await update.message.reply_text(response.choices[0].message.content)
        except Exception as e:
            await update.message.reply_text(f"Error in chat: {e}")

    async def show_analysis(self, update: Update, context):
        try:
            with open('analysis.json', 'r') as f:
                data = json.load(f)
            message = self.format_results(data)
            await update.message.reply_text(message)
        except FileNotFoundError:
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

    def run(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
        self.app.add_handler(MessageHandler(Filters.text & ~Filters.command, self.chat))
        self.app.run_polling()

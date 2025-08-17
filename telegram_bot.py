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

# Eksik sınıfları ve fonksiyonları içe aktar
from mexc_client import MEXCClient  # MEXCClient sınıfını içe aktar
from deepseek_client import DeepSeekClient  # DeepSeekClient sınıfını içe aktar
from storage import Storage  # Storage sınıfını içe aktar
from indicators import calculate_indicators  # calculate_indicators fonksiyonunu içe aktar

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TelegramBot:
    def __init__(self):
        """Telegram botunu başlatır ve gerekli ayarları yapar."""
        logger.debug("TelegramBot başlatılıyor")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            logger.error("TELEGRAM_BOT_TOKEN ortam değişkeni eksik")
            raise ValueError("TELEGRAM_BOT_TOKEN ortam değişkeni eksik")
        
        try:
            # Telegram Application oluştur
            self.app = Application.builder().token(bot_token).build()
            logger.debug("Application başarıyla başlatıldı")
            
            # Komut ve callback handler'ları ekle
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CallbackQueryHandler(self.button))
            self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
            self.web_app = None
            
            # Job queue kontrolü
            if self.app.job_queue is not None:
                try:
                    self.app.job_queue.start()
                    logger.debug("Job queue başarıyla başlatıldı")
                except Exception as e:
                    logger.warning(f"Job queue başlatılamadı: {e}. Job queue olmadan devam ediliyor.")
            else:
                logger.warning("Job queue mevcut değil. Job queue olmadan devam ediliyor.")
        except Exception as e:
            logger.error(f"Application başlatılırken hata: {e}")
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ /start komutu için menü oluşturur. """
        logger.debug(f"Start komutu alındı, chat_id={update.effective_chat.id}")
        keyboard = [
            [InlineKeyboardButton("Top 100 Spot Analizi", callback_data='top_100_spot')],
            [InlineKeyboardButton("Top 300 Spot Analizi", callback_data='top_300_spot')],
            [InlineKeyboardButton("Top 100 Vadeli Analizi", callback_data='top_100_futures')],
            [InlineKeyboardButton("Top 300 Vadeli Analizi", callback_data='top_300_futures')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Analiz için butonları kullanabilirsiniz:", reply_markup=reply_markup)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton tıklamalarını işler ve analiz başlatır."""
        query = update.callback_query
        await query.answer()
        logger.debug(f"Buton tıklandı: {query.data}")
        try:
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            await query.message.reply_text(f"{trade_type.upper()} analizi yapılıyor (Top {limit})...")
            data = {'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type}
            
            # Job queue kontrolü
            if self.app.job_queue is None:
                logger.warning("Job queue mevcut değil, analyze_and_send direkt çalıştırılıyor")
                await self.analyze_and_send(context, data)
            else:
                self.app.job_queue.run_once(
                    self.analyze_and_send,
                    0,
                    data=data,
                    chat_id=self.group_id
                )
        except Exception as e:
            logger.error(f"Buton işleyicisinde hata: {e}")
            await query.message.reply_text(f"Hata: {str(e)}")

    async def analyze_and_send(self, context: ContextTypes.DEFAULT_TYPE, data=None):
        """Coin analizini yapar ve sonuçları gönderir."""
        if data is None:
            data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.debug(f"Analiz yapılıyor: chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"Top {limit} {trade_type} analizi için anlamlı sonuç bulunamadı.")
            logger.info(f"Top {limit} {trade_type} için analiz tamamlandı")
        except Exception as e:
            logger.error(f"analyze_and_send sırasında hata: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"{trade_type} analizi sırasında hata: {str(e)}")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Kullanıcı mesajlarına yanıt verir."""
        logger.debug(f"Mesaj alındı: {update.message.text}")
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
            logger.error(f"Chat sırasında hata: {e}")
            await update.message.reply_text(f"Hata: {str(e)}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Kaydedilmiş analiz sonuçlarını gösterir."""
        logger.debug("show_analysis komutu alındı")
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
            logger.error(f"Analiz yüklenirken hata: {e}")
            await update.message.reply_text(f"Hata: {str(e)}")

    def format_results(self, coin_data, trade_type, symbol):
        """Analiz sonuçlarını biçimlendirir."""
        logger.debug(f"{symbol} ({trade_type}) için sonuçlar biçimlendiriliyor")
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
            logger.error(f"{symbol} için sonuçlar biçimlendirilirken hata: {e}")
            return f"{symbol} analizi biçimlendirilirken hata: {str(e)}"

    async def process_coin(self, symbol, mexc, deepseek, trade_type, chat_id):
        """Tek bir coin için analiz yapar."""
        logger.debug(f"{symbol} ({trade_type}) işleniyor")
        try:
            data = await mexc.fetch_and_save_market_data(symbol)
            if not data:
                logger.warning(f"{symbol} ({trade_type}) için geçerli piyasa verisi yok")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için geçerli piyasa verisi yok")
                return None

            data['indicators'] = calculate_indicators(
                data['klines'].get('1m', []), data['klines'].get('5m', []), data['klines'].get('15m', []),
                data['klines'].get('30m', []), data['klines'].get('60m', []), data.get('order_book')
            )
            if not data['indicators']:
                logger.warning(f"{symbol} ({trade_type}) için gösterge hesaplanamadı")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için gösterge hesaplanamadı")
                return None

            data['deepseek_analysis'] = deepseek.analyze_coin(data, trade_type)
            logger.info(f"{symbol} ({trade_type}) işlendi: fiyat={data.get('price')}, "
                       f"klines_60m={len(data['klines'].get('60m', []))}")

            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"{symbol} ({trade_type}) için analiz gönderildi")

            storage = Storage()
            storage.save_analysis({f'{symbol}_{trade_type}': [data]})

            return data
        except Exception as e:
            logger.error(f"{symbol} ({trade_type}) işlenirken hata: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} işlenirken hata: {str(e)}")
            return None

    async def analyze_coins(self, limit, trade_type, chat_id):
        """Top coin'ler için analiz yapar."""
        logger.debug(f"analyze_coins başlatılıyor: limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        deepseek = DeepSeekClient()

        coins = await mexc.get_top_coins(limit)
        logger.info(f"{len(coins)} coin analiz ediliyor: {coins[:5]}...")

        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_{limit}_{trade_type}': []}
        for symbol in coins:
            coin_data = await self.process_coin(symbol, mexc, deepseek, trade_type, chat_id)
            if coin_data:
                results[f'top_{limit}_{trade_type}'].append(coin_data)
            await asyncio.sleep(3.0)  # Rate limit için bekleme
        logger.info(f"Top {limit} {trade_type} için {len(results[f'top_{limit}_{trade_type}'])} geçerli coin işlendi")
        await mexc.close()
        return results

    async def webhook_handler(self, request):
        """Webhook isteklerini işler."""
        logger.debug("Webhook isteği alındı")
        try:
            raw_data = await request.json()
            update = Update.de_json(raw_data, self.app.bot)
            if not update:
                logger.warning("Geçersiz webhook güncellemesi")
                return web.Response(text="HATA: Geçersiz güncelleme", status=400)
            await self.app.process_update(update)
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Webhook hatası: {e}")
            return web.Response(text=f"HATA: {str(e)}", status=500)

    async def run(self):
        """Webhook sunucusunu başlatır."""
        logger.debug("Webhook sunucusu başlatılıyor")
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        try:
            await self.app.bot.set_webhook(url=webhook_url)
            logger.debug(f"Webhook {webhook_url} adresine ayarlandı")
        except Exception as e:
            logger.error(f"Webhook ayarlanırken hata: {e}")
            raise
        runner = web.AppRunner(self.web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
        await site.start()
        logger.info(f"Webhook sunucusu {os.getenv('PORT', 8443)} portunda çalışıyor")
        await asyncio.Event().wait()

def main():
    """Ana fonksiyon, botu başlatır."""
    logger.debug("Main başlatılıyor")
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    logger.debug("Script başlatıldı")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_coins":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        trade_type = sys.argv[3] if len(sys.argv) > 3 else 'spot'
        bot = TelegramBot()
        asyncio.run(bot.analyze_coins(limit, trade_type, bot.group_id))
    else:
        main()

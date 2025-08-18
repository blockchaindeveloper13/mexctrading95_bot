import os
import pandas as pd
import pandas_ta as ta
import logging
import asyncio
import aiohttp
import signal
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from openai import AsyncOpenAI
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import re
import numpy as np
import json

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Seçilen coinler
COINS = [
    "OKBUSDT", "ADAUSDT", "DOTUSDT", "XLMUSDT", "LTCUSDT",
    "UNIUSDT", "ATOMUSDT", "CRVUSDT", "TRUMPUSDT", "AAVEUSDT", "BNBUSDT"
]

# Konuşma durumları
ASKING_ANALYSIS = 0

class MEXCClient:
    """MEXC Spot API ile iletişim kurar."""
    def __init__(self):
        self.spot_url = "https://api.mexc.com"
        self.session = None

    async def initialize(self):
        """Session’ı başlatır."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def fetch_market_data(self, symbol):
        """Spot piyasası verisi çeker."""
        await self.initialize()
        async with self.session:
            klines = {}
            for interval in ['5m', '15m', '60m']:
                url = f"{self.spot_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit=200"
                async with self.session.get(url) as response:
                    response_data = await response.json() if response.status == 200 else []
                    klines[interval] = {'data': response_data}
                    logger.info(f"Kline response for {symbol} ({interval}): {response_data[:1]}...")
                await asyncio.sleep(0.2)

            order_book_url = f"{self.spot_url}/api/v3/depth?symbol={symbol}&limit=10"
            async with self.session.get(order_book_url) as response:
                order_book = await response.json() if response.status == 200 else {'bids': [], 'asks': []}
                logger.info(f"Order book response for {symbol}: {order_book}")
            await asyncio.sleep(0.2)

            ticker_url = f"{self.spot_url}/api/v3/ticker/price?symbol={symbol}"
            async with self.session.get(ticker_url) as response:
                ticker = await response.json() if response.status == 200 else {'price': '0.0'}
                logger.info(f"Ticker response for {symbol}: {ticker}")
            await asyncio.sleep(0.2)

            ticker_24hr_url = f"{self.spot_url}/api/v3/ticker/24hr?symbol={symbol}"
            async with self.session.get(ticker_24hr_url) as response:
                ticker_24hr = await response.json() if response.status == 200 else {'priceChangePercent': '0.0'}
                logger.info(f"24hr ticker response for {symbol}: {ticker_24hr}")
            await asyncio.sleep(0.2)

            btc_data = await self.fetch_btc_data()
            return {
                'klines': klines,
                'order_book': order_book,
                'price': float(ticker.get('price', 0.0)),
                'funding_rate': 0.0,  # Spot'ta fonlama oranı yok
                'price_change_24hr': float(ticker_24hr.get('priceChangePercent', 0.0)),
                'btc_data': btc_data
            }

    async def fetch_btc_data(self):
        """BTC/USDT spot verilerini çeker."""
        await self.initialize()
        async with self.session:
            url = f"{self.spot_url}/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=200"
            async with self.session.get(url) as response:
                response_data = await response.json() if response.status == 200 else []
                logger.info(f"BTC data response: {response_data[:1]}...")
                return {'data': response_data}

    async def validate_symbol(self, symbol):
        """Sembolü spot piyasasında doğrular."""
        await self.initialize()
        async with self.session:
            url = f"{self.spot_url}/api/v3/ticker/price?symbol={symbol}"
            async with self.session.get(url) as response:
                response_data = await response.json()
                logger.info(f"Validate symbol response for {symbol}: {response_data}")
                return response.status == 200 and 'price' in response_data

    async def close(self):
        """Session’ı kapatır."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

class DeepSeekClient:
    """DeepSeek API ile analiz yapar."""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    async def analyze_coin(self, symbol, data):
        """Coin için long/short analizi yapar."""
        support_levels = data['indicators'].get('support_levels', [0.0, 0.0, 0.0])
        resistance_levels = data['indicators'].get('resistance_levels', [0.0, 0.0, 0.0])
        prompt = f"""
        {symbol} için vadeli işlem analizi yap (spot piyasa verilerine dayalı). Yanıt tamamen Türkçe, 500-1000 karakter. Verilere dayanarak giriş fiyatı, take-profit (çıkış), stop-loss, kaldıraç, risk/ödül oranı ve trend tahmini üret. ATR > %5 veya BTC korelasyonu > 0.8 ise yatırımdan uzak dur uyarısı ekle. Spot verilerini vadeli işlem için uyarla. Doğal ve profesyonel üslup kullan. Markdown (** vb.) kullanma, sadece emoji kullan. Giriş, take-profit ve stop-loss’u nasıl belirlediğini, hangi göstergelere dayandığını ve analiz sürecini yorumda açıkla. Her alan için tam bir sayı veya metin zorunlu.

        - Mevcut Fiyat: {data['price']} USDT
        - 24 Saatlik Değişim: {data.get('price_change_24hr', 0.0)}%
        - Göstergeler:
          - MA (5m): 50={data['indicators']['ma_5m']['ma50']:.2f}, 200={data['indicators']['ma_5m']['ma200']:.2f}
          - RSI (5m): {data['indicators']['rsi_5m']:.2f}
          - ATR (5m): %{data['indicators']['atr_5m']:.2f}
          - BTC Korelasyonu: {data['indicators']['btc_correlation']:.2f}
        - Destek: {', '.join([f'${x:.2f}' for x in support_levels])}
        - Direnç: {', '.join([f'${x:.2f}' for x in resistance_levels])}

        Çıktı formatı:
        📊 {symbol} Vadeli Analiz ({datetime.now().strftime('%Y-%m-%d %H:%M')})
        🔄 Zaman Dilimleri: 5m, 15m, 1h
        📈 Long Pozisyon:
        - Giriş: $X
        - Take-Profit: $Y
        - Stop-Loss: $Z
        - Kaldıraç: Nx
        - Risk/Ödül: A:B
        - Trend: [Yükseliş/Düşüş/Nötr]
        📉 Short Pozisyon:
        - Giriş: $X
        - Take-Profit: $Y
        - Stop-Loss: $Z
        - Kaldıraç: Nx
        - Risk/Ödül: A:B
        - Trend: [Yükseliş/Düşüş/Nötr]
        📍 Destek: {', '.join([f'${x:.2f}' for x in support_levels])}
        📍 Direnç: {', '.join([f'${x:.2f}' for x in resistance_levels])}
        ⚠️ Volatilite: %{data['indicators']['atr_5m']:.2f} ({'Yüksek, uzak dur!' if data['indicators']['atr_5m'] > 5 else 'Normal'})
        🔗 BTC Korelasyonu: {data['indicators']['btc_correlation']:.2f} ({'Yüksek, dikkat!' if data['indicators']['btc_correlation'] > 0.8 else 'Normal'})
        💬 Yorum: [Analiz süreci, hangi göstergelere dayandığı, giriş/take-profit/stop-loss seçim gerekçesi]
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    stream=False
                ),
                timeout=120.0
            )
            analysis_text = response.choices[0].message.content
            logger.info(f"DeepSeek raw response for {symbol}: {analysis_text}")
            if len(analysis_text) < 500:
                analysis_text += " " * (500 - len(analysis_text))
            
            required_fields = ['Giriş', 'Take-Profit', 'Stop-Loss', 'Kaldıraç', 'Risk/Ödül', 'Trend', 'Yorum']
            missing_fields = []
            for field in required_fields:
                if field == 'Yorum':
                    if not analysis_text.strip() or '💬 Yorum:' not in analysis_text:
                        missing_fields.append(field)
                elif field not in analysis_text:
                    missing_fields.append(field)
            if missing_fields:
                raise ValueError(f"DeepSeek yanıtı eksik: {', '.join(missing_fields)}")

            return {'analysis_text': analysis_text}
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.error(f"DeepSeek API error for {symbol}: {e}")
            raise Exception(f"DeepSeek API'den veri alınamadı: {str(e)}")

class Storage:
    """Analizleri SQLite’ta depolar."""
    def __init__(self):
        self.db_path = "analysis.db"
        self.init_db()

    def init_db(self):
        """SQLite veritabanını başlatır."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp TEXT,
                    indicators TEXT,
                    analysis_text TEXT
                )
            """)
            conn.commit()

    def save_analysis(self, symbol, data):
        """Analizi kaydeder."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analyses (symbol, timestamp, indicators, analysis_text)
                    VALUES (?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    json.dumps(data['indicators']),
                    data['deepseek_analysis']['analysis_text']
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite error while saving analysis for {symbol}: {e}")

    def get_previous_analysis(self, symbol):
        """Sembol için en son analizi çeker."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM analyses WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1
                """, (symbol,))
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching analysis for {symbol}: {e}")
            return None

    def get_latest_analysis(self, symbol):
        """Sembol için en son analizi çeker (sohbet için)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT analysis_text FROM analyses WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1
                """, (symbol,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching latest analysis for {symbol}: {e}")
            return None

class TelegramBot:
    """Telegram botu."""
    def __init__(self):
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.storage = Storage()
        self.mexc = MEXCClient()
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.app = Application.builder().token(bot_token).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(ConversationHandler(
            entry_points=[MessageHandler(filters.Regex(r'(?i)(analiz|trend|long|short|destek|direnç|yorum|neden).*\b(' + '|'.join(COINS) + r')\b'), self.handle_analysis_query)],
            states={
                ASKING_ANALYSIS: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_analysis_query)]
            },
            fallbacks=[CommandHandler("cancel", self.cancel)]
        ))
        self.active_analyses = {}
        self.shutdown_event = asyncio.Event()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Coin butonlarını gösterir."""
        keyboard = [[InlineKeyboardButton(coin, callback_data=f"analyze_{coin}")] for coin in COINS]
        await update.message.reply_text("📈 Vadeli işlem analizi için coin seç:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton tıklamalarını işler."""
        query = update.callback_query
        try:
            await query.answer()
        except Exception as e:
            logger.error(f"Error answering callback query: {e}")
        symbol = query.data.replace("analyze_", "")
        analysis_key = f"{symbol}_futures_{update.effective_chat.id}"
        if analysis_key in self.active_analyses:
            await query.message.reply_text(f"⏳ {symbol} için analiz yapılıyor, bekleyin.")
            return
        self.active_analyses[analysis_key] = True
        try:
            if not await self.mexc.validate_symbol(symbol):
                await query.message.reply_text(f"❌ Hata: {symbol} spot piyasasında mevcut değil.")
                return
            await query.message.reply_text(f"🔍 {symbol} için vadeli işlem analizi yapılıyor...")
            task = self.process_coin(symbol, update.effective_chat.id)
            if task is not None:
                asyncio.create_task(task)
        finally:
            del self.active_analyses[analysis_key]

    async def handle_analysis_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analizle ilgili soruları yanıtlar."""
        text = update.message.text.lower()
        logger.info(f"Received message: {text}")
        symbol = None
        for coin in COINS:
            if coin.lower() in text:
                symbol = coin
                break
        if not symbol:
            logger.info(f"No valid coin symbol found in message: {text}")
            await update.message.reply_text("🔍 Lütfen geçerli bir coin sembolü belirtin (örn: ADAUSDT).")
            return ASKING_ANALYSIS

        current_analysis = self.storage.get_latest_analysis(symbol)
        previous_analysis = self.storage.get_previous_analysis(symbol)
        response = f"📊 {symbol} için analiz:\n"

        if "trend" in text:
            if current_analysis:
                long_trend = re.search(r'📈 Long Pozisyon:.*?Trend: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                short_trend = re.search(r'📉 Short Pozisyon:.*?Trend: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                response += f"📈 Güncel Long Trend: {long_trend.group(1) if long_trend else 'Bilinmiyor'}\n"
                response += f"📉 Güncel Short Trend: {short_trend.group(1) if short_trend else 'Bilinmiyor'}\n"
            if previous_analysis:
                prev_long_trend = re.search(r'📈 Long Pozisyon:.*?Trend: (.*?)(?:\n|$)', previous_analysis['analysis_text'], re.DOTALL)
                prev_short_trend = re.search(r'📉 Short Pozisyon:.*?Trend: (.*?)(?:\n|$)', previous_analysis['analysis_text'], re.DOTALL)
                response += f"📅 Geçmiş ({previous_analysis['timestamp']}):\n"
                response += f"📈 Geçmiş Long Trend: {prev_long_trend.group(1) if prev_long_trend else 'Bilinmiyor'}\n"
                response += f"📉 Geçmiş Short Trend: {prev_short_trend.group(1) if prev_short_trend else 'Bilinmiyor'}\n"
                if current_analysis and prev_long_trend and prev_short_trend:
                    response += f"🔄 Karşılaştırma: Long trend {'değişmedi' if long_trend and long_trend.group(1) == prev_long_trend.group(1) else 'değişti'}, Short trend {'değişmedi' if short_trend and short_trend.group(1) == prev_short_trend.group(1) else 'değişti'}.\n"

        if "long" in text:
            if current_analysis:
                long_match = re.search(r'📈 Long Pozisyon:(.*?)(?:📉|$)', current_analysis, re.DOTALL)
                response += f"📈 Güncel Long Pozisyon:\n{long_match.group(1).strip() if long_match else 'Bilinmiyor'}\n"
            if previous_analysis:
                prev_long_match = re.search(r'📈 Long Pozisyon:(.*?)(?:📉|$)', previous_analysis['analysis_text'], re.DOTALL)
                response += f"📅 Geçmiş Long Pozisyon ({previous_analysis['timestamp']}):\n{prev_long_match.group(1).strip() if prev_long_match else 'Bilinmiyor'}\n"
                if current_analysis and prev_long_match:
                    curr_entry = re.search(r'Giriş: \$([\d.]+)', long_match.group(1)) if long_match else None
                    prev_entry = re.search(r'Giriş: \$([\d.]+)', prev_long_match.group(1)) if prev_long_match else None
                    if curr_entry and prev_entry:
                        response += f"🔄 Karşılaştırma: Giriş fiyatı ${prev_entry.group(1)}’den ${curr_entry.group(1)}’e {'yükseldi' if float(curr_entry.group(1)) > float(prev_entry.group(1)) else 'düştü'}.\n"

        if "short" in text:
            if current_analysis:
                short_match = re.search(r'📉 Short Pozisyon:(.*?)(?:💬|$)', current_analysis, re.DOTALL)
                response += f"📉 Güncel Short Pozisyon:\n{short_match.group(1).strip() if short_match else 'Bilinmiyor'}\n"
            if previous_analysis:
                prev_short_match = re.search(r'📉 Short Pozisyon:(.*?)(?:💬|$)', previous_analysis['analysis_text'], re.DOTALL)
                response += f"📅 Geçmiş Short Pozisyon ({previous_analysis['timestamp']}):\n{prev_short_match.group(1).strip() if prev_short_match else 'Bilinmiyor'}\n"
                if current_analysis and prev_short_match:
                    curr_entry = re.search(r'Giriş: \$([\d.]+)', short_match.group(1)) if short_match else None
                    prev_entry = re.search(r'Giriş: \$([\d.]+)', prev_short_match.group(1)) if prev_short_match else None
                    if curr_entry and prev_entry:
                        response += f"🔄 Karşılaştırma: Giriş fiyatı ${prev_entry.group(1)}’den ${curr_entry.group(1)}’e {'yükseldi' if float(curr_entry.group(1)) > float(prev_entry.group(1)) else 'düştü'}.\n"

        if "destek" in text:
            if current_analysis:
                support_match = re.search(r'📍 Destek: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                response += f"📍 Güncel Destek: {support_match.group(1) if support_match else 'Bilinmiyor'}\n"
            if previous_analysis:
                support_levels = json.loads(previous_analysis['indicators'])['support_levels']
                response += f"📅 Geçmiş Destek ({previous_analysis['timestamp']}): {', '.join([f'${x:.2f}' for x in support_levels])}\n"

        if "direnç" in text:
            if current_analysis:
                resistance_match = re.search(r'📍 Direnç: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                response += f"📍 Güncel Direnç: {resistance_match.group(1) if resistance_match else 'Bilinmiyor'}\n"
            if previous_analysis:
                resistance_levels = json.loads(previous_analysis['indicators'])['resistance_levels']
                response += f"📅 Geçmiş Direnç ({previous_analysis['timestamp']}): {', '.join([f'${x:.2f}' for x in resistance_levels])}\n"

        if "yorum" in text or "neden" in text:
            if current_analysis:
                comment_match = re.search(r'💬 Yorum:(.*)', current_analysis, re.DOTALL)
                response += f"💬 Güncel Yorum: {comment_match.group(1).strip()[:500] if comment_match else 'Bilinmiyor'}\n"
            if previous_analysis:
                comment_match = re.search(r'💬 Yorum:(.*)', previous_analysis['analysis_text'], re.DOTALL)
                response += f"📅 Geçmiş Yorum ({previous_analysis['timestamp']}): {comment_match.group(1).strip()[:500] if comment_match else 'Bilinmiyor'}\n"

        if not current_analysis and not previous_analysis:
            response += f"❌ {symbol} için analiz bulunamadı. Yeni analiz yapmamı ister misiniz? (örn: /analyze_{symbol})"

        logger.info(f"Sending response for {symbol}: {response[:200]}...")
        await update.message.reply_text(response)
        return ASKING_ANALYSIS

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Konuşmayı iptal eder."""
        await update.message.reply_text("❌ Konuşma iptal edildi.")
        return ConversationHandler.END

    async def process_coin(self, symbol, chat_id):
        """Coin için analiz yapar."""
        try:
            data = await self.mexc.fetch_market_data(symbol)
            if not data or not any(data.get('klines', {}).get(interval, {}).get('data') for interval in ['5m', '15m', '60m']):
                await self.app.bot.send_message(chat_id=chat_id, text=f"❌ {symbol} için veri yok.")
                return
            data['indicators'] = calculate_indicators(data['klines'], data['order_book'], data['btc_data'], symbol)
            deepseek = DeepSeekClient()
            data['deepseek_analysis'] = await deepseek.analyze_coin(symbol, data)
            message = data['deepseek_analysis']['analysis_text']
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            self.storage.save_analysis(symbol, data)
            return data
        except Exception as e:
            logger.error(f"Error processing coin {symbol}: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"❌ {symbol} analizi sırasında hata: {str(e)}")
            return

    async def run(self):
        """Webhook sunucusunu başlatır."""
        try:
            logger.info("Starting application...")
            await self.mexc.initialize()
            await self.app.initialize()
            await self.app.start()
            web_app = web.Application()
            web_app.router.add_post('/webhook', self.webhook_handler)
            webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
            await self.app.bot.set_webhook(url=webhook_url)
            runner = web.AppRunner(web_app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8443)))
            await site.start()
            logger.info("Application started successfully")
            await self.shutdown_event.wait()
        except Exception as e:
            logger.error(f"Error starting application: {e}")
        finally:
            logger.info("Shutting down application...")
            await self.mexc.close()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Application shut down")

    async def webhook_handler(self, request):
        """Webhook isteklerini işler."""
        try:
            raw_data = await request.json()
            update = Update.de_json(raw_data, self.app.bot)
            if update:
                await self.app.process_update(update)
            return web.Response(text="OK")
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return web.Response(text="Error", status=500)

def calculate_indicators(kline_data, order_book, btc_data, symbol):
    """Teknik göstergeleri hesaplar."""
    indicators = {}
    for interval in ['5m', '15m', '60m']:
        kline = kline_data.get(interval, {}).get('data', [])
        if kline and len(kline) > 1:
            try:
                df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['volume'] = df['volume'].astype(float)

                sma_50 = ta.sma(df['close'], length=50) if len(df) >= 50 else None
                sma_200 = ta.sma(df['close'], length=200) if len(df) >= 200 else None
                indicators[f'ma_{interval}'] = {
                    'ma50': sma_50.iloc[-1] if sma_50 is not None and not sma_50.empty else 0.0,
                    'ma200': sma_200.iloc[-1] if sma_200 is not None and not sma_200.empty else 0.0
                }

                rsi = ta.rsi(df['close'], length=14) if len(df) >= 14 else None
                indicators[f'rsi_{interval}'] = rsi.iloc[-1] if rsi is not None and not rsi.empty else 0.0

                atr = ta.atr(df['high'], df['low'], df['close'], length=14) if len(df) >= 14 else None
                indicators[f'atr_{interval}'] = (atr.iloc[-1] / df['close'].iloc[-1] * 100) if atr is not None and not atr.empty else 0.0

                last_row = df.iloc[-1]
                pivot = (last_row['high'] + last_row['low'] + last_row['close']) / 3
                range_high_low = last_row['high'] - last_row['low']
                indicators['support_levels'] = [
                    pivot - range_high_low * 0.5,
                    pivot - range_high_low * 0.618,
                    pivot - range_high_low
                ]
                indicators['resistance_levels'] = [
                    pivot + range_high_low * 0.5,
                    pivot + range_high_low * 0.618,
                    pivot + range_high_low
                ]
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol} ({interval}): {e}")
                indicators.update({
                    f'ma_{interval}': {'ma50': 0.0, 'ma200': 0.0},
                    f'rsi_{interval}': 0.0,
                    f'atr_{interval}': 0.0,
                    'support_levels': [0.0, 0.0, 0.0],
                    'resistance_levels': [0.0, 0.0, 0.0]
                })
        else:
            logger.warning(f"No kline data for {symbol} ({interval})")
            indicators.update({
                f'ma_{interval}': {'ma50': 0.0, 'ma200': 0.0},
                f'rsi_{interval}': 0.0,
                f'atr_{interval}': 0.0,
                'support_levels': [0.0, 0.0, 0.0],
                'resistance_levels': [0.0, 0.0, 0.0]
            })

    if order_book.get('bids') and order_book.get('asks'):
        bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
        ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        indicators['bid_ask_ratio'] = bid_volume / ask_volume if ask_volume > 0 else 0.0
    else:
        indicators['bid_ask_ratio'] = 0.0
        logger.warning(f"Order book for {symbol} has no bids or asks")

    if btc_data.get('data') and len(btc_data['data']) > 1:
        btc_df = pd.DataFrame(btc_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
        btc_df['close'] = btc_df['close'].astype(float)
        if kline_data.get('5m', {}).get('data') and len(kline_data['5m']['data']) > 1:
            coin_df = pd.DataFrame(kline_data['5m']['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
            coin_df['close'] = coin_df['close'].astype(float)
            correlation = coin_df['close'].corr(btc_df['close'])
            indicators['btc_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            indicators['btc_correlation'] = 0.0
    else:
        indicators['btc_correlation'] = 0.0

    return indicators

def main():
    bot = TelegramBot()

    def handle_sigterm(*args):
        logger.info("Received SIGTERM, shutting down...")
        bot.shutdown_event.set()
        asyncio.create_task(bot.mexc.close())
        asyncio.create_task(bot.app.stop())
        asyncio.create_task(bot.app.shutdown())

    signal.signal(signal.SIGTERM, handle_sigterm)
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()

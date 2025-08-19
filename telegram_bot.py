import os
import pandas as pd
import pandas_ta as ta
import logging
import asyncio
import aiohttp
import signal
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from openai import AsyncOpenAI
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime, timedelta
import sqlite3
import re
import numpy as np
import json

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Seçilen coinler ve kısaltmaları
COINS = {
    "OKBUSDT": ["okb", "okbusdt"],
    "ADAUSDT": ["ada", "adausdt"],
    "DOTUSDT": ["dot", "dotusdt"],
    "XLMUSDT": ["xlm", "xlmusdt"],
    "LTCUSDT": ["ltc", "ltcusdt"],
    "UNIUSDT": ["uni", "uniusdt"],
    "ATOMUSDT": ["atom", "atomusdt"],
    "CRVUSDT": ["crv", "crvusdt"],
    "TRUMPUSDT": ["trump", "trumpusdt"],
    "AAVEUSDT": ["aave", "aaveusdt"],
    "BNBUSDT": ["bnb", "bnbusdt"],
    "ETHUSDT": ["eth", "ethusdt", "ethereum"],
    "BTCUSDT": ["btc", "btcusdt", "bitcoin"],
    "LINKUSDT": ["link", "linkusdt", "chainlink"],
    "MKRUSDT": ["mkr", "mkrusdt", "maker"]
}

class CoinMarketCapClient:
    """CoinMarketCap API ile iletişim kurar."""
    def __init__(self):
        self.base_url = "https://pro-api.coinmarketcap.com"
        self.api_key = os.getenv('COINMARKETCAP_API_KEY')
        self.session = None

    async def initialize(self):
        """Session’ı başlatır."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={'X-CMC_PRO_API_KEY': self.api_key})

    async def fetch_kline_data(self, symbol, interval, count=50):
        """CoinMarketCap’ten kline verisi çeker."""
        await self.initialize()
        try:
            cmc_intervals = {'6h': '6h', '12h': '12h', '1w': 'weekly'}
            if interval not in cmc_intervals:
                return {'data': []}
            url = f"{self.base_url}/v2/cryptocurrency/ohlcv/historical?symbol={symbol[:-4]}&interval={cmc_intervals[interval]}&count={count}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if response_data['data'].get(symbol[:-4]):
                        data = [
                            [quote['timestamp'], quote['open'], quote['high'], quote['low'], quote['close'], quote['volume'], quote['timestamp'], quote['quote_volume']]
                            for quote in response_data['data'][symbol[:-4]][0]['quotes']
                        ]
                        logger.info(f"CoinMarketCap kline response for {symbol} ({interval}): {data[:1]}...")
                        return {'data': data}
                    else:
                        logger.warning(f"No CoinMarketCap kline data for {symbol} ({interval})")
                        return {'data': []}
                else:
                    logger.error(f"Failed to fetch CoinMarketCap kline data for {symbol} ({interval}): {response.status}")
                    return {'data': []}
        except Exception as e:
            logger.error(f"Error fetching CoinMarketCap kline data for {symbol} ({interval}): {e}")
            return {'data': []}
        finally:
            await asyncio.sleep(0.5)

    async def close(self):
        """Session’ı kapatır."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

class MEXCClient:
    """MEXC Spot API ile iletişim kurar."""
    def __init__(self):
        self.spot_url = "https://api.mexc.com"
        self.session = None
        self.cmc_client = CoinMarketCapClient()

    async def initialize(self):
        """Session’ı başlatır."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        await self.cmc_client.initialize()

    async def fetch_market_data(self, symbol):
        """Spot piyasası verisi çeker."""
        await self.initialize()
        try:
            klines = {}
            mexc_intervals = ['5m', '15m', '60m', '1d']
            for interval in mexc_intervals:
                url = f"{self.spot_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if response_data and isinstance(response_data, list) and len(response_data) > 0:
                            klines[interval] = {'data': response_data}
                            logger.info(f"Kline response for {symbol} ({interval}): {response_data[:1]}...")
                        else:
                            logger.warning(f"No valid kline data for {symbol} ({interval})")
                            klines[interval] = {'data': []}
                    else:
                        logger.error(f"Failed to fetch kline data for {symbol} ({interval}): {response.status}")
                        klines[interval] = {'data': []}
                    await asyncio.sleep(0.5)

            # CoinMarketCap’ten 6h, 12h, 1w verilerini çek
            cmc_intervals = ['6h', '12h', '1w']
            for interval in cmc_intervals:
                klines[interval] = await self.cmc_client.fetch_kline_data(symbol, interval)

            order_book_url = f"{self.spot_url}/api/v3/depth?symbol={symbol}&limit=10"
            async with self.session.get(order_book_url) as response:
                order_book = await response.json() if response.status == 200 else {'bids': [], 'asks': []}
                logger.info(f"Order book response for {symbol}: {order_book}")
            await asyncio.sleep(0.5)

            ticker_url = f"{self.spot_url}/api/v3/ticker/price?symbol={symbol}"
            async with self.session.get(ticker_url) as response:
                ticker = await response.json() if response.status == 200 else {'price': '0.0'}
                logger.info(f"Ticker response for {symbol}: {ticker}")
            await asyncio.sleep(0.5)

            ticker_24hr_url = f"{self.spot_url}/api/v3/ticker/24hr?symbol={symbol}"
            async with self.session.get(ticker_24hr_url) as response:
                ticker_24hr = await response.json() if response.status == 200 else {'priceChangePercent': '0.0'}
                logger.info(f"24hr ticker response for {symbol}: {ticker_24hr}")
            await asyncio.sleep(0.5)

            btc_data = await self.fetch_btc_data()
            eth_data = await self.fetch_eth_data()
            return {
                'klines': klines,
                'order_book': order_book,
                'price': float(ticker.get('price', 0.0)),
                'funding_rate': 0.0,  # Spot'ta fonlama oranı yok
                'price_change_24hr': float(ticker_24hr.get('priceChangePercent', 0.0)),
                'btc_data': btc_data,
                'eth_data': eth_data
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
        finally:
            await self.close()

    async def fetch_btc_data(self):
        """BTC/USDT spot verilerini çeker."""
        await self.initialize()
        async with self.session:
            url = f"{self.spot_url}/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=50"
            async with self.session.get(url) as response:
                response_data = await response.json() if response.status == 200 else []
                logger.info(f"BTC data response: {response_data[:1]}...")
                return {'data': response_data}

    async def fetch_eth_data(self):
        """ETH/USDT spot verilerini çeker."""
        await self.initialize()
        async with self.session:
            url = f"{self.spot_url}/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=50"
            async with self.session.get(url) as response:
                response_data = await response.json() if response.status == 200 else []
                logger.info(f"ETH data response: {response_data[:1]}...")
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
        await self.cmc_client.close()

class DeepSeekClient:
    """DeepSeek API ile analiz yapar ve doğal dil işleme sağlar."""
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    async def analyze_coin(self, symbol, data):
        """Coin için long/short analizi yapar."""
        support_levels = data['indicators'].get('support_levels', [0.0, 0.0, 0.0])
        resistance_levels = data['indicators'].get('resistance_levels', [0.0, 0.0, 0.0])
        fib_levels = data['indicators'].get('fibonacci_levels', [0.0, 0.0, 0.0, 0.0, 0.0])
        ichimoku = data['indicators'].get('ichimoku_1d', {})
        prompt = f"""
        {symbol} için vadeli işlem analizi yap (spot piyasa verilerine dayalı). Yanıt tamamen Türkçe, 500-1000 karakter. Verilere dayanarak giriş fiyatı, take-profit, stop-loss, kaldıraç, risk/ödül oranı ve trend tahmini üret. ATR > %5 veya BTC/ETH korelasyonu > 0.8 ise yatırımdan uzak dur uyarısı ekle, ancak teorik long ve short pozisyon parametrelerini sağla. Spot verilerini vadeli işlem için uyarla. Doğal ve profesyonel üslup kullan. Markdown (** vb.) kullanma, sadece emoji kullan. Giriş, take-profit ve stop-loss’u nasıl belirlediğini, hangi göstergelere dayandığını ve analiz sürecini yorumda açıkla. Kısa vadeli (5m, 15m, 60m) ve uzun vadeli (6h, 12h, 1d, 1w) trendleri ayrı ayrı değerlendir. Uzun vadeli veriler CoinMarketCap’ten alındı, eksiklikleri belirt. Her alan için tam bir sayı veya metin zorunlu.

        - Mevcut Fiyat: {data['price']} USDT
        - 24 Saatlik Değişim: {data.get('price_change_24hr', 0.0)}%
        - Kısa Vadeli Göstergeler (5m):
          - MA: 50={data['indicators']['ma_5m']['ma50']:.2f}, 200={data['indicators']['ma_5m']['ma200']:.2f}
          - RSI: {data['indicators']['rsi_5m']:.2f}
          - ATR: %{data['indicators']['atr_5m']:.2f}
          - MACD: {data['indicators']['macd_5m']['macd']:.2f}, Sinyal: {data['indicators']['macd_5m']['signal']:.2f}
          - Bollinger: Üst={data['indicators']['bbands_5m']['upper']:.2f}, Alt={data['indicators']['bbands_5m']['lower']:.2f}
          - Stochastic: %K={data['indicators']['stoch_5m']['k']:.2f}, %D={data['indicators']['stoch_5m']['d']:.2f}
          - OBV: {data['indicators']['obv_5m']:.2f}
        - Uzun Vadeli Göstergeler (1d):
          - MA: 50={data['indicators']['ma_1d']['ma50']:.2f}, 200={data['indicators']['ma_1d']['ma200']:.2f}
          - RSI: {data['indicators']['rsi_1d']:.2f}
          - ATR: %{data['indicators']['atr_1d']:.2f}
          - MACD: {data['indicators']['macd_1d']['macd']:.2f}, Sinyal: {data['indicators']['macd_1d']['signal']:.2f}
          - Ichimoku: Tenkan={ichimoku.get('tenkan', 0.0):.2f}, Kijun={ichimoku.get('kijun', 0.0):.2f}, Bulut={ichimoku.get('senkou_a', 0.0):.2f}/{ichimoku.get('senkou_b', 0.0):.2f}
        - Fibonacci Seviyeleri (1d): {', '.join([f'${x:.2f}' for x in fib_levels])}
        - Destek: {', '.join([f'${x:.2f}' for x in support_levels])}
        - Direnç: {', '.join([f'${x:.2f}' for x in resistance_levels])}
        - BTC Korelasyonu: {data['indicators']['btc_correlation']:.2f}
        - ETH Korelasyonu: {data['indicators']['eth_correlation']:.2f}

        Çıktı formatı:
        📊 {symbol} Vadeli Analiz ({datetime.now().strftime('%Y-%m-%d %H:%M')})
        🔄 Zaman Dilimleri: 5m, 15m, 60m, 6h, 12h, 1d, 1w
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
        📍 Fibonacci: {', '.join([f'${x:.2f}' for x in fib_levels])}
        ⚠️ Volatilite: %{data['indicators']['atr_5m']:.2f} ({'Yüksek, uzak dur!' if data['indicators']['atr_5m'] > 5 else 'Normal'})
        🔗 BTC Korelasyonu: {data['indicators']['btc_correlation']:.2f} ({'Yüksek, dikkat!' if data['indicators']['btc_correlation'] > 0.8 else 'Normal'})
        🔗 ETH Korelasyonu: {data['indicators']['eth_correlation']:.2f} ({'Yüksek, dikkat!' if data['indicators']['eth_correlation'] > 0.8 else 'Normal'})
        💬 Yorum: [Kısa vadeli (5m, 15m, 60m) MEXC’ten, uzun vadeli (6h, 12h, 1w) CoinMarketCap’ten alındı. MACD, Bollinger, Stochastic, OBV ve Ichimoku’ya dayalı giriş/take-profit/stop-loss seçim gerekçesi. Yüksek korelasyon veya volatilite varsa neden yatırımdan uzak durulmalı açıkla. Uzun vadeli veri eksikse, kısa vadeli verilere odaklan ve eksikliği belirt.]
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    stream=False
                ),
                timeout=180.0
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

    async def generate_natural_response(self, user_message, context_info, symbol=None):
        """Doğal dil yanıtı üretir."""
        prompt = f"""
        Türkçe, ultra samimi ve esprili bir şekilde yanıt ver. Kullanıcıya 'kanka' diye hitap et, hafif argo kullan ama abartma. Mesajına uygun, akıcı ve doğal bir muhabbet kur. Eğer sembol ({symbol}) varsa, bağlama uygun şekilde atıfta bulun ve BTC/ETH korelasyonlarını vurgula. Konuşma geçmişini ve son analizi dikkate al. Maksimum 200 karakter. Hata mesajlarından kaçın, her zaman muhabbeti devam ettir!

        Kullanıcı mesajı: {user_message}
        Bağlam: {context_info}
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    stream=False
                ),
                timeout=60.0
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(f"DeepSeek API timeout for natural response")
            return "😂 Kanka, internet nazlandı, bi’ daha sor bakalım!"
        except Exception as e:
            logger.error(f"DeepSeek natural response error: {e}")
            return "😅 Kanka, neyi kastediyosun, bi’ açar mısın? Hadi, muhabbet edelim!"

class Storage:
    """Analizleri ve konuşma geçmişini SQLite’ta depolar."""
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    user_message TEXT,
                    bot_response TEXT,
                    timestamp TEXT,
                    symbol TEXT
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

    def save_conversation(self, chat_id, user_message, bot_response, symbol=None):
        """Konuşma geçmişini kaydeder."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO conversations (chat_id, user_message, bot_response, timestamp, symbol)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chat_id,
                    user_message,
                    bot_response,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    symbol
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite error while saving conversation for chat_id {chat_id}: {e}")

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

    def get_conversation_history(self, chat_id, limit=10):
        """Sohbet geçmişini çeker."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_message, bot_response, timestamp, symbol 
                    FROM conversations 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (chat_id, limit))
                results = cursor.fetchall()
                return [{'user_message': row[0], 'bot_response': row[1], 'timestamp': row[2], 'symbol': row[3]} for row in results]
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching conversation history for chat_id {chat_id}: {e}")
            return []

    def get_last_symbol(self, chat_id):
        """Son konuşmada kullanılan sembolü çeker."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol FROM conversations 
                    WHERE chat_id = ? AND symbol IS NOT NULL 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (chat_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching last symbol for chat_id {chat_id}: {e}")
            return None

    def clean_old_data(self, days=7):
        """Eski verileri temizler."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM analyses WHERE timestamp < ?", 
                              (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
                cursor.execute("DELETE FROM conversations WHERE timestamp < ?", 
                              (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
                conn.commit()
                logger.info("Old data cleaned from SQLite")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while cleaning old data: {e}")

class TelegramBot:
    """Telegram botu."""
    def __init__(self):
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.storage = Storage()
        self.mexc = MEXCClient()
        self.deepseek = DeepSeekClient()
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.app = Application.builder().token(bot_token).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        self.active_analyses = {}
        self.shutdown_event = asyncio.Event()
        self.is_running = False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Coin butonlarını gösterir."""
        keyboard = [
            [InlineKeyboardButton("BTCUSDT", callback_data="analyze_BTCUSDT"), InlineKeyboardButton("ETHUSDT", callback_data="analyze_ETHUSDT")],
            *[[InlineKeyboardButton(coin, callback_data=f"analyze_{coin}")] for coin in COINS.keys() if coin not in ["BTCUSDT", "ETHUSDT"]]
        ]
        response = (
            "📈 Kanka, hadi bakalım! Coin analizi mi yapalım, yoksa başka muhabbet mi çevirelim? 😎\n"
            "Örnek: 'ADA analiz', 'nasılsın', 'geçmiş', 'BTC korelasyonu'.\n"
        )
        await update.message.reply_text(response, reply_markup=InlineKeyboardMarkup(keyboard))
        self.storage.save_conversation(update.effective_chat.id, update.message.text, response)

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
            response = f"⏳ Kanka, {symbol} için analiz yapıyorum, az sabret! 😅"
            await query.message.reply_text(response)
            self.storage.save_conversation(update.effective_chat.id, query.data, response, symbol)
            return
        self.active_analyses[analysis_key] = True
        try:
            if not await self.mexc.validate_symbol(symbol):
                response = f"😓 Kanka, {symbol} piyasada yok gibi. Başka coin mi bakalım?"
                await query.message.reply_text(response)
                self.storage.save_conversation(update.effective_chat.id, query.data, response, symbol)
                return
            response = f"🔍 {symbol} için analiz yapıyorum, hemen geliyor! 🚀"
            await query.message.reply_text(response)
            self.storage.save_conversation(update.effective_chat.id, query.data, response, symbol)
            task = self.process_coin(symbol, update.effective_chat.id)
            if task is not None:
                asyncio.create_task(task)
        finally:
            del self.active_analyses[analysis_key]

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tüm metin mesajlarını işler ve doğal yanıtlar üretir."""
        text = update.message.text.lower()
        chat_id = update.effective_chat.id
        logger.info(f"Received message: {text}")

        # Konuşma geçmişini al
        history = self.storage.get_conversation_history(chat_id, limit=3)
        context_info = f"Son konuşmalar: {history}"

        # Geçmiş konuşmaları gösterme
        if "geçmiş" in text:
            history = self.storage.get_conversation_history(chat_id, limit=10)
            if not history:
                response = "📜 Kanka, henüz muhabbet geçmişimiz yok. Hadi başlayalım! 😎"
            else:
                response = "📜 Son muhabbetler:\n"
                for entry in history:
                    response += f"🕒 {entry['timestamp']}\n👤 Sen: {entry['user_message']}\n🤖 Ben: {entry['bot_response']}\n"
                    if entry['symbol']:
                        response += f"💱 Coin: {entry['symbol']}\n"
                    response += "\n"
            await update.message.reply_text(response)
            self.storage.save_conversation(chat_id, text, response)
            return

        # Coin sembolünü bul
        symbol = None
        for coin, aliases in COINS.items():
            if any(alias in text for alias in aliases):
                symbol = coin
                break

        # Eğer sembol bulunmadıysa, son konuşmadan sembolü çek
        if not symbol:
            symbol = self.storage.get_last_symbol(chat_id)
            if symbol:
                logger.info(f"Using last symbol {symbol} from conversation history")

        # Anahtar kelimeler
        keywords = ['analiz', 'trend', 'long', 'short', 'destek', 'direnç', 'yorum', 'neden', 'korelasyon']
        matched_keyword = next((k for k in keywords if k in text), None)

        # Doğal dil yanıtı için bağlam
        context_info += f"\nSon {symbol} analizi: {self.storage.get_latest_analysis(symbol) or 'Yok' if symbol else 'Yok'}"

        # Analiz istenirse
        if matched_keyword == 'analiz' and symbol:
            analysis_key = f"{symbol}_futures_{chat_id}"
            if analysis_key in self.active_analyses:
                response = f"⏳ Kanka, {symbol} için analiz yapıyorum, az bekle! 😅"
                await update.message.reply_text(response)
                self.storage.save_conversation(chat_id, text, response, symbol)
                return
            self.active_analyses[analysis_key] = True
            try:
                if not await self.mexc.validate_symbol(symbol):
                    response = f"😓 Kanka, {symbol} piyasada yok gibi. Başka coin mi bakalım?"
                    await update.message.reply_text(response)
                    self.storage.save_conversation(chat_id, text, response, symbol)
                    return
                response = f"🔍 {symbol} için analiz yapıyorum, hemen geliyor! 🚀"
                await update.message.reply_text(response)
                self.storage.save_conversation(chat_id, text, response, symbol)
                task = self.process_coin(symbol, chat_id)
                if task is not None:
                    asyncio.create_task(task)
            finally:
                del self.active_analyses[analysis_key]
            return

        # Korelasyon sorusu
        if matched_keyword == 'korelasyon' and symbol:
            current_analysis = self.storage.get_latest_analysis(symbol)
            response = await self.deepseek.generate_natural_response(text, context_info, symbol)
            if current_analysis:
                btc_corr = re.search(r'🔗 BTC Korelasyonu: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                eth_corr = re.search(r'🔗 ETH Korelasyonu: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                response += f"\n🔗 BTC Korelasyon: {btc_corr.group(1) if btc_corr else 'Bilinmiyor'}"
                response += f"\n🔗 ETH Korelasyon: {eth_corr.group(1) if eth_corr else 'Bilinmiyor'}"
            else:
                response += f"\n😅 Kanka, {symbol} için analiz yok. Hemen yapayım mı? (örn: {symbol} analiz)"
            await update.message.reply_text(response)
            self.storage.save_conversation(chat_id, text, response, symbol)
            return

        # Diğer anahtar kelimeler veya sembol varsa
        if symbol and matched_keyword:
            current_analysis = self.storage.get_latest_analysis(symbol)
            response = await self.deepseek.generate_natural_response(text, context_info, symbol)
            if current_analysis:
                if matched_keyword == 'trend':
                    long_trend = re.search(r'📈 Long Pozisyon:.*?Trend: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                    short_trend = re.search(r'📉 Short Pozisyon:.*?Trend: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                    response += f"\n📈 Long: {long_trend.group(1) if long_trend else 'Bilinmiyor'}\n📉 Short: {short_trend.group(1) if short_trend else 'Bilinmiyor'}"
                elif matched_keyword == 'long':
                    long_match = re.search(r'📈 Long Pozisyon:(.*?)(?:📉|$)', current_analysis, re.DOTALL)
                    response += f"\n📈 Long: {long_match.group(1).strip() if long_match else 'Bilinmiyor'}"
                elif matched_keyword == 'short':
                    short_match = re.search(r'📉 Short Pozisyon:(.*?)(?:💬|$)', current_analysis, re.DOTALL)
                    response += f"\n📉 Short: {short_match.group(1).strip() if short_match else 'Bilinmiyor'}"
                elif matched_keyword == 'destek':
                    support_match = re.search(r'📍 Destek: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                    response += f"\n📍 Destek: {support_match.group(1) if support_match else 'Bilinmiyor'}"
                elif matched_keyword == 'direnç':
                    resistance_match = re.search(r'📍 Direnç: (.*?)(?:\n|$)', current_analysis, re.DOTALL)
                    response += f"\n📍 Direnç: {resistance_match.group(1) if resistance_match else 'Bilinmiyor'}"
                elif matched_keyword in ['yorum', 'neden']:
                    comment_match = re.search(r'💬 Yorum:(.*)', current_analysis, re.DOTALL)
                    response += f"\n💬 Yorum: {comment_match.group(1).strip()[:500] if comment_match else 'Bilinmiyor'}"
            else:
                response += f"\n😅 Kanka, {symbol} için analiz yok. Hemen yapayım mı? (örn: {symbol} analiz)"
            await update.message.reply_text(response)
            self.storage.save_conversation(chat_id, text, response, symbol)
            return

        # Genel sohbet için doğal yanıt
        response = await self.deepseek.generate_natural_response(text, context_info, symbol)
        await update.message.reply_text(response)
        self.storage.save_conversation(chat_id, text, response, symbol)

    async def process_coin(self, symbol, chat_id):
        """Coin için analiz yapar."""
        try:
            data = await self.mexc.fetch_market_data(symbol)
            if not data or not any(data.get('klines', {}).get(interval, {}).get('data') for interval in ['5m', '15m', '60m', '6h', '12h', '1d', '1w']):
                response = f"😓 Kanka, {symbol} için veri bulamadım. Başka coin mi bakalım?"
                await self.app.bot.send_message(chat_id=chat_id, text=response)
                self.storage.save_conversation(chat_id, symbol, response, symbol)
                return
            data['indicators'] = calculate_indicators(data['klines'], data['order_book'], data['btc_data'], data['eth_data'], symbol)
            data['deepseek_analysis'] = await self.deepseek.analyze_coin(symbol, data)
            message = data['deepseek_analysis']['analysis_text']
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            self.storage.save_analysis(symbol, data)
            self.storage.save_conversation(chat_id, symbol, message, symbol)
            self.storage.clean_old_data()  # Bellek optimizasyonu
            return data
        except Exception as e:
            logger.error(f"Error processing coin {symbol}: {e}")
            response = f"😅 Kanka, {symbol} analizi yaparken bi’ şeyler ters gitti. Tekrar deneyelim mi?"
            await self.app.bot.send_message(chat_id=chat_id, text=response)
            self.storage.save_conversation(chat_id, symbol, response, symbol)
            return

    async def run(self):
        """Webhook sunucusunu başlatır."""
        try:
            logger.info("Starting application...")
            self.is_running = True
            await self.mexc.initialize()
            await self.app.initialize()
            await self.app.start()
            webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
            # Webhook durumunu kontrol et
            current_webhook = await self.app.bot.get_webhook_info()
            if current_webhook.url != webhook_url:
                logger.info(f"Setting new webhook: {webhook_url}")
                await self.app.bot.set_webhook(url=webhook_url)
            else:
                logger.info("Webhook already set, skipping...")
            web_app = web.Application()
            web_app.router.add_post('/webhook', self.webhook_handler)
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
            if self.is_running:
                try:
                    await self.app.stop()
                    await self.app.shutdown()
                    logger.info("Webhook preserved to avoid re-setting")
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}")
                self.is_running = False
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

def calculate_indicators(kline_data, order_book, btc_data, eth_data, symbol):
    """Teknik göstergeleri hesaplar."""
    indicators = {}
    fallback_interval = None
    for interval in ['5m', '15m', '60m', '6h', '12h', '1d', '1w']:
        kline = kline_data.get(interval, {}).get('data', [])
        if kline and len(kline) > 1:
            fallback_interval = interval
            try:
                df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
                df['close'] = df['close'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['volume'] = df['volume'].astype(float)

                # Hareketli Ortalamalar
                sma_50 = ta.sma(df['close'], length=50) if len(df) >= 50 else None
                sma_200 = ta.sma(df['close'], length=200) if len(df) >= 200 else None
                indicators[f'ma_{interval}'] = {
                    'ma50': sma_50.iloc[-1] if sma_50 is not None and not sma_50.empty else 0.0,
                    'ma200': sma_200.iloc[-1] if sma_200 is not None and not sma_200.empty else 0.0
                }

                # RSI
                rsi = ta.rsi(df['close'], length=14) if len(df) >= 14 else None
                indicators[f'rsi_{interval}'] = rsi.iloc[-1] if rsi is not None and not rsi.empty else 0.0

                # ATR
                atr = ta.atr(df['high'], df['low'], df['close'], length=14) if len(df) >= 14 else None
                indicators[f'atr_{interval}'] = (atr.iloc[-1] / df['close'].iloc[-1] * 100) if atr is not None and not atr.empty else 0.0

                # MACD
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9) if len(df) >= 26 else None
                indicators[f'macd_{interval}'] = {
                    'macd': macd['MACD_12_26_9'].iloc[-1] if macd is not None and not macd.empty else 0.0,
                    'signal': macd['MACDs_12_26_9'].iloc[-1] if macd is not None and not macd.empty else 0.0
                }

                # Bollinger Bantları
                bbands = ta.bbands(df['close'], length=20, std=2) if len(df) >= 20 else None
                indicators[f'bbands_{interval}'] = {
                    'upper': bbands['BBU_20_2.0'].iloc[-1] if bbands is not None and not bbands.empty else 0.0,
                    'lower': bbands['BBL_20_2.0'].iloc[-1] if bbands is not None and not bbands.empty else 0.0
                }

                # Stochastic Oscillator
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3) if len(df) >= 14 else None
                indicators[f'stoch_{interval}'] = {
                    'k': stoch['STOCHk_14_3_3'].iloc[-1] if stoch is not None and not stoch.empty else 0.0,
                    'd': stoch['STOCHd_14_3_3'].iloc[-1] if stoch is not None and not stoch.empty else 0.0
                }

                # OBV
                obv = ta.obv(df['close'], df['volume']) if len(df) >= 1 else None
                indicators[f'obv_{interval}'] = obv.iloc[-1] if obv is not None and not obv.empty else 0.0

                # Destek ve Direnç
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

                # Fibonacci Retracement
                if len(df) >= 50:
                    high = df['high'].tail(50).max()
                    low = df['low'].tail(50).min()
                    diff = high - low
                    indicators['fibonacci_levels'] = [
                        low + diff * 0.236,
                        low + diff * 0.382,
                        low + diff * 0.5,
                        low + diff * 0.618,
                        low + diff * 0.786
                    ]
                else:
                    indicators['fibonacci_levels'] = [0.0, 0.0, 0.0, 0.0, 0.0]

                # Ichimoku Cloud
                if interval == '1d' and len(df) >= 52:
                    try:
                        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)[0]
                        indicators[f'ichimoku_{interval}'] = {
                            'tenkan': ichimoku['ITS_9'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                            'kijun': ichimoku['ITK_26'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                            'senkou_a': ichimoku['ISA_9'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                            'senkou_b': ichimoku['ISB_26'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0
                        }
                    except Exception as e:
                        logger.error(f"Ichimoku error for {symbol} ({interval}): {e}")
                        # 60m’den Ichimoku türet
                        if '60m' in kline_data and kline_data['60m'].get('data'):
                            df_60m = pd.DataFrame(kline_data['60m']['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
                            df_60m['close'] = df_60m['close'].astype(float)
                            df_60m['high'] = df_60m['high'].astype(float)
                            df_60m['low'] = df_60m['low'].astype(float)
                            if len(df_60m) >= 52:
                                try:
                                    ichimoku = ta.ichimoku(df_60m['high'], df_60m['low'], df_60m['close'], tenkan=9, kijun=26, senkou=52)[0]
                                    indicators[f'ichimoku_{interval}'] = {
                                        'tenkan': ichimoku['ITS_9'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                                        'kijun': ichimoku['ITK_26'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                                        'senkou_a': ichimoku['ISA_9'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0,
                                        'senkou_b': ichimoku['ISB_26'].iloc[-1] if ichimoku is not None and not ichimoku.empty else 0.0
                                    }
                                except Exception as e:
                                    logger.error(f"Fallback Ichimoku error for {symbol} (60m): {e}")
                                    indicators[f'ichimoku_{interval}'] = {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}
                            else:
                                indicators[f'ichimoku_{interval}'] = {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}
                        else:
                            indicators[f'ichimoku_{interval}'] = {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}
                else:
                    indicators[f'ichimoku_{interval}'] = {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}

            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol} ({interval}): {e}")
                indicators.update({
                    f'ma_{interval}': {'ma50': 0.0, 'ma200': 0.0},
                    f'rsi_{interval}': 0.0,
                    f'atr_{interval}': 0.0,
                    f'macd_{interval}': {'macd': 0.0, 'signal': 0.0},
                    f'bbands_{interval}': {'upper': 0.0, 'lower': 0.0},
                    f'stoch_{interval}': {'k': 0.0, 'd': 0.0},
                    f'obv_{interval}': 0.0,
                    'support_levels': [0.0, 0.0, 0.0],
                    'resistance_levels': [0.0, 0.0, 0.0],
                    'fibonacci_levels': [0.0, 0.0, 0.0, 0.0, 0.0],
                    f'ichimoku_{interval}': {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}
                })
        else:
            logger.warning(f"No kline data for {symbol} ({interval})")
            indicators.update({
                f'ma_{interval}': {'ma50': 0.0, 'ma200': 0.0},
                f'rsi_{interval}': 0.0,
                f'atr_{interval}': 0.0,
                f'macd_{interval}': {'macd': 0.0, 'signal': 0.0},
                f'bbands_{interval}': {'upper': 0.0, 'lower': 0.0},
                f'stoch_{interval}': {'k': 0.0, 'd': 0.0},
                f'obv_{interval}': 0.0,
                'support_levels': [0.0, 0.0, 0.0],
                'resistance_levels': [0.0, 0.0, 0.0],
                'fibonacci_levels': [0.0, 0.0, 0.0, 0.0, 0.0],
                f'ichimoku_{interval}': {'tenkan': 0.0, 'kijun': 0.0, 'senkou_a': 0.0, 'senkou_b': 0.0}
            })

    # Destek/Direnç ve Fibonacci için yedek hesaplama (60m’den)
    if all(indicators['support_levels'][i] == 0.0 for i in range(3)) and fallback_interval:
        kline = kline_data.get(fallback_interval, {}).get('data', [])
        if kline and len(kline) > 1:
            df = pd.DataFrame(kline, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
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
            if len(df) >= 50:
                high = df['high'].tail(50).max()
                low = df['low'].tail(50).min()
                diff = high - low
                indicators['fibonacci_levels'] = [
                    low + diff * 0.236,
                    low + diff * 0.382,
                    low + diff * 0.5,
                    low + diff * 0.618,
                    low + diff * 0.786
                ]

    if order_book.get('bids') and order_book.get('asks'):
        bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
        ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        indicators['bid_ask_ratio'] = bid_volume / ask_volume if ask_volume > 0 else 0.0
    else:
        indicators['bid_ask_ratio'] = 0.0
        logger.warning(f"Order book for {symbol} has no bids or asks")

    # BTC korelasyonu
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

    # ETH korelasyonu
    if eth_data.get('data') and len(eth_data['data']) > 1:
        eth_df = pd.DataFrame(eth_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
        eth_df['close'] = eth_df['close'].astype(float)
        if kline_data.get('5m', {}).get('data') and len(kline_data['5m']['data']) > 1:
            coin_df = pd.DataFrame(kline_data['5m']['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume'])
            coin_df['close'] = coin_df['close'].astype(float)
            correlation = coin_df['close'].corr(eth_df['close'])
            indicators['eth_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            indicators['eth_correlation'] = 0.0
    else:
        indicators['eth_correlation'] = 0.0

    return indicators

def main():
    bot = TelegramBot()

    def handle_sigterm(*args):
        logger.info("Received SIGTERM, shutting down...")
        bot.shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()

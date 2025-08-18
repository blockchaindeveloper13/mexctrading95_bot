import os
import pandas as pd
import pandas_ta as ta
import logging
import asyncio
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, MessageHandler, filters
from openai import OpenAI
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

    async def fetch_market_data(self, symbol):
        """Spot piyasası verisi çeker."""
        async with aiohttp.ClientSession() as session:
            klines = {}
            for interval in ['5m', '15m', '60m']:
                url = f"{self.spot_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit=200"
                async with session.get(url) as response:
                    response_data = await response.json() if response.status == 200 else []
                    klines[interval] = {'data': response_data}
                    logger.info(f"Kline response for {symbol} ({interval}): {response_data[:1]}...")
                await asyncio.sleep(0.2)

            order_book_url = f"{self.spot_url}/api/v3/depth?symbol={symbol}&limit=10"
            async with session.get(order_book_url) as response:
                order_book = await response.json() if response.status == 200 else {'bids': [], 'asks': []}
                logger.info(f"Order book response for {symbol}: {order_book}")
            await asyncio.sleep(0.2)

            ticker_url = f"{self.spot_url}/api/v3/ticker/price?symbol={symbol}"
            async with session.get(ticker_url) as response:
                ticker = await response.json() if response.status == 200 else {'price': '0.0'}
                logger.info(f"Ticker response for {symbol}: {ticker}")
            await asyncio.sleep(0.2)

            ticker_24hr_url = f"{self.spot_url}/api/v3/ticker/24hr?symbol={symbol}"
            async with session.get(ticker_24hr_url) as response:
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
        async with aiohttp.ClientSession() as session:
            url = f"{self.spot_url}/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=200"
            async with session.get(url) as response:
                response_data = await response.json() if response.status == 200 else []
                logger.info(f"BTC data response: {response_data[:1]}...")
                return {'data': response_data}

    async def validate_symbol(self, symbol):
        """Sembolü spot piyasasında doğrular."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.spot_url}/api/v3/ticker/price?symbol={symbol}"
            async with session.get(url) as response:
                response_data = await response.json()
                logger.info(f"Validate symbol response for {symbol}: {response_data}")
                return response.status == 200 and 'price' in response_data

class DeepSeekClient:
    """DeepSeek API ile analiz yapar."""
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    async def analyze_coin(self, symbol, data):
        """Coin için long/short analizi yapar."""
        support_levels = data['indicators'].get('support_levels', [0.0, 0.0, 0.0])
        resistance_levels = data['indicators'].get('resistance_levels', [0.0, 0.0, 0.0])
        prompt = f"""
        {symbol} için vadeli işlem analizi yap (spot piyasa verilerine dayalı). Yanıt tamamen Türkçe, 500-2000 karakter. Aşağıdaki verilere dayanarak giriş, çıkış, stop-loss, kaldıraç, risk/ödül oranı ve trend tahmini üret. ATR > %5 veya BTC korelasyonu > 0.8 ise yatırımdan uzak dur uyarısı ekle. Spot verilerini vadeli işlem için uyarla. Doğal ve profesyonel üslup kullan. Markdown (** vb.) kullanma, sadece emoji kullan.

        - Mevcut Fiyat: {data['price']} USDT
        - 24 Saatlik Değişim: {data.get('price_change_24hr', 0.0)}%
        - Göstergeler:
          - MA (5m): 50={data['indicators']['ma_5m']['ma50']:.2f}, 200={data['indicators']['ma_5m']['ma200']:.2f}
          - EMA (5m): 12={data['indicators']['ema_5m']['ema12']:.2f}, 26={data['indicators']['ema_5m']['ema26']:.2f}
          - SAR (5m): {data['indicators']['sar_5m']:.2f}
          - Bollinger (5m): Üst={data['indicators']['bb_5m']['upper']:.2f}, Alt={data['indicators']['bb_5m']['lower']:.2f}
          - MACD (5m): {data['indicators']['macd_5m']:.2f}
          - KDJ (5m): {data['indicators']['kdj_5m']:.2f}
          - RSI (5m): {data['indicators']['rsi_5m']:.2f}
          - StochRSI (5m): {data['indicators']['stochrsi_5m']:.2f}
          - ATR (5m): %{data['indicators']['atr_5m']:.2f}
          - BTC Korelasyonu: {data['indicators']['btc_correlation']:.2f}
        - Destek: {', '.join([f'${x:.2f}' for x in support_levels])}
        - Direnç: {', '.join([f'${x:.2f}' for x in resistance_levels])}

        Çıktı formatı:
        📈 Long Pozisyon:
        - Giriş: $X
        - Çıkış: $Y
        - Stop-Loss: $Z
        - Kaldıraç: Nx
        - Risk/Ödül: A:B
        - Trend: [Yükseliş/Düşüş/Nötr]
        📉 Short Pozisyon:
        - Giriş: $X
        - Çıkış: $Y
        - Stop-Loss: $Z
        - Kaldıraç: Nx
        - Risk/Ödül: A:B
        - Trend: [Yükseliş/Düşüş/Nötr]
        💬 Yorum: [Detaylı analiz ve gerekçe]
        """
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    stream=False
                ),
                timeout=10.0
            )
            analysis_text = response.choices[0].message.content
            if len(analysis_text) < 500:
                analysis_text += " " * (500 - len(analysis_text))
            return {
                'long': self.parse_response(analysis_text, data['price'], 'long'),
                'short': self.parse_response(analysis_text, data['price'], 'short'),
                'comment': analysis_text
            }
        except asyncio.TimeoutError:
            logger.error(f"DeepSeek API timeout for {symbol}")
            return {
                'long': {'entry_price': data['price'], 'exit_price': data['price'] * 1.02, 'stop_loss': data['price'] * 0.98, 'leverage': '3x', 'risk_reward_ratio': 1.0, 'trend': 'Nötr'},
                'short': {'entry_price': data['price'], 'exit_price': data['price'] * 0.98, 'stop_loss': data['price'] * 1.02, 'leverage': '3x', 'risk_reward_ratio': 1.0, 'trend': 'Nötr'},
                'comment': f"Analiz yapılamadı: DeepSeek API zaman aşımına uğradı."
            }
        except Exception as e:
            logger.error(f"DeepSeek API error for {symbol}: {e}")
            return {
                'long': {'entry_price': data['price'], 'exit_price': data['price'] * 1.02, 'stop_loss': data['price'] * 0.98, 'leverage': '3x', 'risk_reward_ratio': 1.0, 'trend': 'Nötr'},
                'short': {'entry_price': data['price'], 'exit_price': data['price'] * 0.98, 'stop_loss': data['price'] * 1.02, 'leverage': '3x', 'risk_reward_ratio': 1.0, 'trend': 'Nötr'},
                'comment': f"Analiz yapılamadı: {str(e)}"
            }

    def parse_response(self, text, current_price, position):
        """DeepSeek yanıtını ayrıştırır."""
        result = {
            'entry_price': current_price,
            'exit_price': current_price * (1.02 if position == 'long' else 0.98),
            'stop_loss': current_price * (0.98 if position == 'long' else 1.02),
            'leverage': '3x',
            'risk_reward_ratio': 1.0,
            'trend': 'Nötr'
        }
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            number_match = re.search(r'\d+\.?\d*', line)
            if f'{position}: giriş' in line and number_match:
                result['entry_price'] = float(number_match.group(0))
            elif f'{position}: çıkış' in line and number_match:
                result['exit_price'] = float(number_match.group(0))
            elif f'{position}: stop-loss' in line and number_match:
                result['stop_loss'] = float(number_match.group(0))
            elif f'{position}: kaldıraç' in line:
                result['leverage'] = line.split(':')[1].strip() if ':' in line else '3x'
            elif f'{position}: risk/ödül' in line and number_match:
                result['risk_reward_ratio'] = float(number_match.group(0))
            elif f'{position}: trend' in line:
                result['trend'] = line.split(':')[1].strip() if ':' in line else 'Nötr'
        return result

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
                    entry_price_long REAL,
                    exit_price_long REAL,
                    stop_loss_long REAL,
                    leverage_long TEXT,
                    entry_price_short REAL,
                    exit_price_short REAL,
                    stop_loss_short REAL,
                    leverage_short TEXT,
                    support_levels TEXT,
                    resistance_levels TEXT,
                    trend_long TEXT,
                    trend_short TEXT,
                    risk_reward_long REAL,
                    risk_reward_short REAL,
                    comment TEXT
                )
            """)
            conn.commit()

    def save_analysis(self, symbol, data):
        """Analizi kaydeder."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analyses (
                        symbol, timestamp, indicators,
                        entry_price_long, exit_price_long, stop_loss_long, leverage_long,
                        entry_price_short, exit_price_short, stop_loss_short, leverage_short,
                        support_levels, resistance_levels, trend_long, trend_short,
                        risk_reward_long, risk_reward_short, comment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    json.dumps(data['indicators']),
                    data['deepseek_analysis']['long']['entry_price'],
                    data['deepseek_analysis']['long']['exit_price'],
                    data['deepseek_analysis']['long']['stop_loss'],
                    data['deepseek_analysis']['long']['leverage'],
                    data['deepseek_analysis']['short']['entry_price'],
                    data['deepseek_analysis']['short']['exit_price'],
                    data['deepseek_analysis']['short']['stop_loss'],
                    data['deepseek_analysis']['short']['leverage'],
                    json.dumps(data['indicators'].get('support_levels', [0.0, 0.0, 0.0])),
                    json.dumps(data['indicators'].get('resistance_levels', [0.0, 0.0, 0.0])),
                    data['deepseek_analysis']['long']['trend'],
                    data['deepseek_analysis']['short']['trend'],
                    data['deepseek_analysis']['long']['risk_reward_ratio'],
                    data['deepseek_analysis']['short']['risk_reward_ratio'],
                    data['deepseek_analysis']['comment']
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

                # MA50 ve MA200
                sma_50 = ta.sma(df['close'], length=50) if len(df) >= 50 else None
                sma_200 = ta.sma(df['close'], length=200) if len(df) >= 200 else None
                indicators[f'ma_{interval}'] = {
                    'ma50': sma_50.iloc[-1] if sma_50 is not None and not sma_50.empty else 0.0,
                    'ma200': sma_200.iloc[-1] if sma_200 is not None and not sma_200.empty else 0.0
                }

                # EMA12 ve EMA26
                ema_12 = ta.ema(df['close'], length=12) if len(df) >= 12 else None
                ema_26 = ta.ema(df['close'], length=26) if len(df) >= 26 else None
                indicators[f'ema_{interval}'] = {
                    'ema12': ema_12.iloc[-1] if ema_12 is not None and not ema_12.empty else 0.0,
                    'ema26': ema_26.iloc[-1] if ema_26 is not None and not ema_26.empty else 0.0
                }

                # SAR
                sar = ta.psar(df['high'], df['low'], df['close']) if len(df) >= 14 else None
                indicators[f'sar_{interval}'] = sar['PSARl_0.02_0.2'].iloc[-1] if sar is not None and not sar.empty else 0.0

                # Bollinger Bands
                bb = ta.bbands(df['close'], length=20, std=2) if len(df) >= 20 else None
                indicators[f'bb_{interval}'] = {
                    'upper': bb['BBU_20_2.0'].iloc[-1] if bb is not None and not bb.empty else 0.0,
                    'lower': bb['BBL_20_2.0'].iloc[-1] if bb is not None and not bb.empty else 0.0
                }

                # MACD
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9) if len(df) >= 26 else None
                indicators[f'macd_{interval}'] = macd['MACD_12_26_9'].iloc[-1] if macd is not None and not macd.empty else 0.0

                # KDJ
                kdj = ta.kdj(df['high'], df['low'], df['close'], length=9) if len(df) >= 9 else None
                indicators[f'kdj_{interval}'] = kdj['K_9_3'].iloc[-1] if kdj is not None and not kdj.empty else 0.0

                # RSI
                rsi = ta.rsi(df['close'], length=14) if len(df) >= 14 else None
                indicators[f'rsi_{interval}'] = rsi.iloc[-1] if rsi is not None and not rsi.empty else 0.0

                # StochRSI
                stochrsi = ta.stochrsi(df['close'], length=14) if len(df) >= 14 else None
                indicators[f'stochrsi_{interval}'] = stochrsi['STOCHRSIk_14_14_3_3'].iloc[-1] if stochrsi is not None and not stochrsi.empty else 0.0

                # ATR
                atr = ta.atr(df['high'], df['low'], df['close'], length=14) if len(df) >= 14 else None
                indicators[f'atr_{interval}'] = (atr.iloc[-1] / df['close'].iloc[-1] * 100) if atr is not None and not atr.empty else 0.0

                # Manuel Pivot Noktaları ve Fibonacci Seviyeleri
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
                    f'ema_{interval}': {'ema12': 0.0, 'ema26': 0.0},
                    f'sar_{interval}': 0.0,
                    f'bb_{interval}': {'upper': 0.0, 'lower': 0.0},
                    f'macd_{interval}': 0.0,
                    f'kdj_{interval}': 0.0,
                    f'rsi_{interval}': 0.0,
                    f'stochrsi_{interval}': 0.0,
                    f'atr_{interval}': 0.0,
                    'support_levels': [0.0, 0.0, 0.0],
                    'resistance_levels': [0.0, 0.0, 0.0]
                })
        else:
            logger.warning(f"No kline data for {symbol} ({interval})")
            indicators.update({
                f'ma_{interval}': {'ma50': 0.0, 'ma200': 0.0},
                f'ema_{interval}': {'ema12': 0.0, 'ema26': 0.0},
                f'sar_{interval}': 0.0,
                f'bb_{interval}': {'upper': 0.0, 'lower': 0.0},
                f'macd_{interval}': 0.0,
                f'kdj_{interval}': 0.0,
                f'rsi_{interval}': 0.0,
                f'stochrsi_{interval}': 0.0,
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

class TelegramBot:
    """Telegram botu."""
    def __init__(self):
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.storage = Storage()
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.app = Application.builder().token(bot_token).build()
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(ConversationHandler(
            entry_points=[MessageHandler(filters.Regex(r'(?i)analiz|trend|long|short|destek|direnç'), self.handle_analysis_query)],
            states={
                ASKING_ANALYSIS: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_analysis_query)]
            },
            fallbacks=[CommandHandler("cancel", self.cancel)]
        ))
        self.active_analyses = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Coin butonlarını gösterir."""
        keyboard = [[InlineKeyboardButton(coin, callback_data=f"analyze_{coin}")] for coin in COINS]
        await update.message.reply_text("📈 Vadeli işlem analizi için coin seç:", reply_markup=InlineKeyboardMarkup(keyboard))

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton tıklamalarını işler."""
        query = update.callback_query
        try:
            await query.answer()  # Hemen yanıt ver
        except Exception as e:
            logger.error(f"Error answering callback query: {e}")
        symbol = query.data.replace("analyze_", "")
        analysis_key = f"{symbol}_futures_{update.effective_chat.id}"
        if analysis_key in self.active_analyses:
            await query.message.reply_text(f"⏳ {symbol} için analiz yapılıyor, bekleyin.")
            return
        self.active_analyses[analysis_key] = True
        mexc = MEXCClient()
        if not await mexc.validate_symbol(symbol):
            await query.message.reply_text(f"❌ Hata: {symbol} spot piyasasında mevcut değil.")
            del self.active_analyses[analysis_key]
            return
        await query.message.reply_text(f"🔍 {symbol} için vadeli işlem analizi yapılıyor...")
        asyncio.create_task(self.process_coin(symbol, mexc, update.effective_chat.id))  # Asenkron görev
        del self.active_analyses[analysis_key]

    async def handle_analysis_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analizle ilgili soruları yanıtlar."""
        text = update.message.text.lower()
        symbol = None
        for coin in COINS:
            if coin.lower() in text:
                symbol = coin
                break
        if not symbol:
            await update.message.reply_text("🔍 Lütfen geçerli bir coin sembolü belirtin (örn: LTCUSDT).")
            return ASKING_ANALYSIS

        analysis = self.storage.get_previous_analysis(symbol)
        if not analysis:
            await update.message.reply_text(f"❌ {symbol} için önceki analiz bulunamadı. Yeni analiz yapmamı ister misiniz?")
            return ASKING_ANALYSIS

        response = f"📊 {symbol} için en son analiz ({analysis['timestamp']}):\n"
        if "trend" in text:
            response += f"📈 Long Trend: {analysis['trend_long']}\n📉 Short Trend: {analysis['trend_short']}\n"
        if "long" in text:
            response += (
                f"📈 Long Pozisyon:\n"
                f"- Giriş: ${analysis['entry_price_long']:.2f}\n"
                f"- Çıkış: ${analysis['exit_price_long']:.2f}\n"
                f"- Stop-Loss: ${analysis['stop_loss_long']:.2f}\n"
                f"- Kaldıraç: {analysis['leverage_long']}\n"
                f"- Risk/Ödül: {analysis['risk_reward_long']:.2f}\n"
            )
        if "short" in text:
            response += (
                f"📉 Short Pozisyon:\n"
                f"- Giriş: ${analysis['entry_price_short']:.2f}\n"
                f"- Çıkış: ${analysis['exit_price_short']:.2f}\n"
                f"- Stop-Loss: ${analysis['stop_loss_short']:.2f}\n"
                f"- Kaldıraç: {analysis['leverage_short']}\n"
                f"- Risk/Ödül: {analysis['risk_reward_short']:.2f}\n"
            )
        if "destek" in text:
            support_levels = json.loads(analysis['support_levels'])
            response += f"📍 Destek: {', '.join([f'${x:.2f}' for x in support_levels])}\n"
        if "direnç" in text:
            resistance_levels = json.loads(analysis['resistance_levels'])
            response += f"📍 Direnç: {', '.join([f'${x:.2f}' for x in resistance_levels])}\n"
        if "yorum" in text or "neden" in text:
            response += f"💬 Yorum: {analysis['comment'][:500]}\n"

        await update.message.reply_text(response)
        return ASKING_ANALYSIS

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Konuşmayı iptal eder."""
        await update.message.reply_text("❌ Konuşma iptal edildi.")
        return ConversationHandler.END

    def format_results(self, coin_data, symbol):
        """Analiz sonuçlarını formatlar."""
        indicators = coin_data.get('indicators', {})
        analysis = coin_data.get('deepseek_analysis', {})
        message = (
            f"📊 {symbol} Vadeli Analiz ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            f"🔄 Zaman Dilimleri: 5m, 15m, 1h\n"
            f"📈 Long Pozisyon:\n"
            f"  - Giriş: ${analysis['long']['entry_price']:.2f}\n"
            f"  - Çıkış: ${analysis['long']['exit_price']:.2f}\n"
            f"  - Stop-Loss: ${analysis['long']['stop_loss']:.2f}\n"
            f"  - Kaldıraç: {analysis['long']['leverage']}\n"
            f"  - Risk/Ödül: {analysis['long']['risk_reward_ratio']:.2f}\n"
            f"  - Trend: {analysis['long']['trend']}\n"
            f"📉 Short Pozisyon:\n"
            f"  - Giriş: ${analysis['short']['entry_price']:.2f}\n"
            f"  - Çıkış: ${analysis['short']['exit_price']:.2f}\n"
            f"  - Stop-Loss: ${analysis['short']['stop_loss']:.2f}\n"
            f"  - Kaldıraç: {analysis['short']['leverage']}\n"
            f"  - Risk/Ödül: {analysis['short']['risk_reward_ratio']:.2f}\n"
            f"  - Trend: {analysis['short']['trend']}\n"
            f"📍 Destek: {', '.join([f'${x:.2f}' for x in indicators.get('support_levels', [0.0, 0.0, 0.0])])}\n"
            f"📍 Direnç: {', '.join([f'${x:.2f}' for x in indicators.get('resistance_levels', [0.0, 0.0, 0.0])])}\n"
            f"⚠️ Volatilite: %{indicators.get('atr_5m', 0.0):.2f} ({'Yüksek, uzak dur!' if indicators.get('atr_5m', 0.0) > 5 else 'Normal'})\n"
            f"🔗 BTC Korelasyonu: {indicators.get('btc_correlation', 0.0):.2f} ({'Yüksek, dikkat!' if indicators.get('btc_correlation', 0.0) > 0.8 else 'Normal'})\n"
            f"💬 Yorum: {analysis['comment'][:500]}"
        )
        return message

    async def process_coin(self, symbol, mexc, chat_id):
        """Coin için analiz yapar."""
        try:
            data = await mexc.fetch_market_data(symbol)
            if not data or not any(data.get('klines', {}).get(interval, {}).get('data') for interval in ['5m', '15m', '60m']):
                await self.app.bot.send_message(chat_id=chat_id, text=f"❌ {symbol} için veri yok.")
                return None

            data['indicators'] = calculate_indicators(data['klines'], data['order_book'], data['btc_data'], symbol)
            deepseek = DeepSeekClient()
            data['deepseek_analysis'] = await deepseek.analyze_coin(symbol, data)
            message = self.format_results(data, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            self.storage.save_analysis(symbol, data)
            return data
        except Exception as e:
            logger.error(f"Error processing coin {symbol}: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"❌ {symbol} analizi sırasında hata: {str(e)}")
            return None

    async def run(self):
        """Webhook sunucusunu başlatır."""
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
        await asyncio.Event().wait()

    async def webhook_handler(self, request):
        """Webhook isteklerini işler."""
        raw_data = await request.json()
        update = Update.de_json(raw_data, self.app.bot)
        if update:
            await self.app.process_update(update)
        return web.Response(text="OK")

def main():
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()

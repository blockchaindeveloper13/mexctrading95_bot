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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# GitHub endpoints.json URL'si
ENDPOINTS_JSON_URL = "https://raw.githubusercontent.com/blockchaindeveloper13/mexctrading95_bot/main/endpoints.json"

class MEXCClient:
    """MEXC API ile iletişim kurar."""
    def __init__(self):
        self.base_url = "https://api.mexc.com"

    async def fetch_and_save_market_data(self, symbol, endpoint=None):
        """Belirtilen sembol için piyasa verisi çeker."""
        logger.info(f"{symbol} için piyasa verisi çekiliyor")
        try:
            async with aiohttp.ClientSession() as session:
                klines = {}
                intervals = ['1m', '5m', '60m', '1d']  # 1 dakika, 5 dakika, 60 dakika, 24 saat
                for interval in intervals:
                    if endpoint and interval == '60m':
                        async with session.get(endpoint) as response:
                            if response.status == 200:
                                klines[interval] = await response.json()
                            else:
                                logger.warning(f"{symbol} için {interval} kline verisi alınamadı: {response.status}")
                                klines[interval] = []
                    else:
                        url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
                        async with session.get(url) as response:
                            if response.status == 200:
                                klines[interval] = await response.json()
                            else:
                                logger.warning(f"{symbol} için {interval} kline verisi alınamadı: {response.status}")
                                klines[interval] = []
                    await asyncio.sleep(0.5)

                order_book_url = f"{self.base_url}/api/v3/depth?symbol={symbol}&limit=10"
                async with session.get(order_book_url) as order_book_response:
                    order_book = await order_book_response.json() if order_book_response.status == 200 else {}
                await asyncio.sleep(0.5)

                ticker_url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(ticker_url) as ticker_response:
                    ticker = await ticker_response.json() if ticker_response.status == 200 else {'price': '0.0'}
                await asyncio.sleep(0.5)

                return {
                    'klines': klines,
                    'order_book': order_book,
                    'price': float(ticker.get('price', 0.0))
                }
        except Exception as e:
            logger.error(f"{symbol} için veri çekilirken hata: {e}")
            return None

    async def validate_symbol(self, symbol):
        """Sembolün geçerli olup olmadığını kontrol eder."""
        logger.info(f"{symbol} sembolü doğrulanıyor")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"{symbol} doğrulanırken hata: {e}")
            return False

    async def get_top_coins(self, limit):
        """En yüksek hacimli coin'leri alır."""
        logger.info(f"Top {limit} coin alınıyor")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/24hr"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        usdt_pairs = [coin for coin in data if coin['symbol'].endswith('USDT')]
                        sorted_coins = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                        return [coin['symbol'] for coin in sorted_coins[:limit]]
                    else:
                        logger.warning("Top coin'ler alınamadı")
                        return []
        except Exception as e:
            logger.error(f"Top coin'ler alınırken hata: {e}")
            return []

    async def close(self):
        """MEXC API bağlantısını kapatır."""
        logger.info("MEXCClient bağlantısı kapatılıyor")
        pass

class DeepSeekClient:
    """DeepSeek API ile coin analizi yapar."""
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
        self.storage = Storage()

    def analyze_coin(self, symbol, data, trade_type, chat_id):
        """DeepSeek API ile coin analizi yapar."""
        logger.info(f"{symbol} için {trade_type} analizi yapılıyor, chat_id={chat_id}")
        try:
            conversations = self.storage.load_conversations()
            group_context = conversations.get(str(chat_id), [])
            context_str = "\n".join([f"[{c['timestamp']}] {c['message']}" for c in group_context])

            prompt = f"""
            {symbol} için {trade_type} işlem stratejisi analizi yap. Yanıt tamamen Türkçe olmalı, minimum 500 karakter ve maksimum 5000 karakter uzunluğunda olmalı. Aşağıdaki piyasa verilerini ve teknik göstergeleri kullanarak özgün, detaylı ve bağlama uygun bir analiz üret. Sabit veya tekrarlayan ifadelerden kaçın, doğal ve profesyonel bir üslup kullan. Grup konuşma geçmişini dikkate alarak analizi kişiselleştir ve geçmişteki sorulara veya yorumlara yanıt ver (örneğin, 'yükselecek mi?'). Tüm zaman dilimlerini (1m, 5m, 60m, 1d) analizine dahil et.

            - Mevcut Fiyat: {data['price']} USDT
            - Hacim Değişimi:
              - 1m: {data.get('indicators', {}).get('volume_change_1m', 'Bilinmiyor')}%
              - 5m: {data.get('indicators', {}).get('volume_change_5m', 'Bilinmiyor')}%
              - 60m: {data.get('indicators', {}).get('volume_change_60m', 'Bilinmiyor')}%
              - 1d: {data.get('indicators', {}).get('volume_change_1d', 'Bilinmiyor')}%
            - RSI:
              - 1m: {data.get('indicators', {}).get('rsi_1m', 'Bilinmiyor')}
              - 5m: {data.get('indicators', {}).get('rsi_5m', 'Bilinmiyor')}
              - 60m: {data.get('indicators', {}).get('rsi_60m', 'Bilinmiyor')}
              - 1d: {data.get('indicators', {}).get('rsi_1d', 'Bilinmiyor')}
            - MACD:
              - 1m: {data.get('indicators', {}).get('macd_1m', 'Bilinmiyor')}
              - 5m: {data.get('indicators', {}).get('macd_5m', 'Bilinmiyor')}
              - 60m: {data.get('indicators', {}).get('macd_60m', 'Bilinmiyor')}
              - 1d: {data.get('indicators', {}).get('macd_1d', 'Bilinmiyor')}
            - Bid/Ask Oranı: {data.get('indicators', {}).get('bid_ask_ratio', 'Bilinmiyor')}

            Grup konuşma geçmişi:
            {context_str if context_str else 'Grup konuşma geçmişi yok.'}

            Analizinde şu bilgileri dahil et, ancak format ve üslup tamamen sana bağlı:
            - Önerilen giriş, çıkış ve stop-loss fiyatları (USDT cinsinden, yalnızca senin belirlediğin değerler kullanılacak).
            - Kaldıraç önerisi (örneğin, spot için 1x, vadeli için uygun bir seviye).
            - Pump ve dump olasılıkları (% cinsinden, göstergelere dayalı).
            - Trend tahmini (yükseliş, düşüş veya nötr).
            - Destek ve direnç seviyeleri (USDT cinsinden).
            - Risk/ödül oranı.
            - Piyasa duyarlılığı, hacim trendleri ve alım/satım baskısını içeren temel analiz (tüm zaman dilimlerini dikkate al).
            - Özgün bir yorum (Al/Sat/Bekle önerisi ve gerekçeleri, yaratıcı ve bağlama uygun).

            Grup konuşmalarındaki sorulara veya yorumlara yanıt ver. Yanıtın akıcı, profesyonel ve en az 500 karakter olsun. Sabit ifadelerden uzak dur, yaratıcı ol.
            """
            response = self.client.chat.completions.create(
                model="deepseek-moe",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000
            )
            analysis_text = response.choices[0].message.content
            logger.info(f"DeepSeek ham yanıtı ({symbol}): {analysis_text}")
            if len(analysis_text) < 500:
                logger.warning(f"DeepSeek yanıtı ({symbol}) 500 karakterden kısa: {len(analysis_text)} karakter")
                analysis_text += " " * (500 - len(analysis_text))
            return {'short_term': self.parse_deepseek_response(analysis_text, data['price'])}
        except Exception as e:
            logger.error(f"{symbol} için DeepSeek analizi sırasında hata: {e}")
            return {
                'short_term': {
                    'entry_price': data['price'],
                    'exit_price': data['price'] * 1.02,
                    'stop_loss': data['price'] * 0.98,
                    'leverage': '1x' if trade_type == 'spot' else '3x',
                    'pump_probability': 50,
                    'dump_probability': 50,
                    'trend': 'Nötr',
                    'support_level': data['price'] * 0.95,
                    'resistance_level': data['price'] * 1.05,
                    'risk_reward_ratio': 1.0,
                    'fundamental_analysis': 'Analiz başarısız: Veri yetersiz.',
                    'comment': self.generate_dynamic_default_comment(data)
                }
            }

    def parse_deepseek_response(self, text, current_price):
        """DeepSeek yanıtını esnek bir şekilde ayrıştırır."""
        try:
            result = {
                'entry_price': None,
                'exit_price': None,
                'stop_loss': None,
                'leverage': '1x',
                'pump_probability': 50,
                'dump_probability': 50,
                'trend': 'Nötr',
                'support_level': None,
                'resistance_level': None,
                'risk_reward_ratio': 1.0,
                'fundamental_analysis': text[:500] if text else 'Analiz başarısız',
                'comment': text if text else 'Analiz başarısız: Veri yetersiz.'
            }

            lines = text.split('\n')
            for line in lines:
                line = line.strip().lower()
                number_match = re.search(r'\d+\.?\d*', line)
                if 'giriş fiyatı' in line and number_match:
                    result['entry_price'] = float(number_match.group(0))
                elif 'çıkış fiyatı' in line and number_match:
                    result['exit_price'] = float(number_match.group(0))
                elif 'stop loss' in line and number_match:
                    result['stop_loss'] = float(number_match.group(0))
                elif 'kaldıraç' in line:
                    result['leverage'] = line.split(':')[1].strip() if ':' in line else '1x'
                elif 'pump olasılığı' in line and re.search(r'\d+', line):
                    result['pump_probability'] = int(re.search(r'\d+', line).group(0))
                elif 'dump olasılığı' in line and re.search(r'\d+', line):
                    result['dump_probability'] = int(re.search(r'\d+', line).group(0))
                elif 'trend' in line:
                    result['trend'] = line.split(':')[1].strip() if ':' in line else 'Nötr'
                elif 'destek seviyesi' in line and number_match:
                    result['support_level'] = float(number_match.group(0))
                elif 'direnç seviyesi' in line and number_match:
                    result['resistance_level'] = float(number_match.group(0))
                elif 'risk/ödül oranı' in line and number_match:
                    result['risk_reward_ratio'] = float(number_match.group(0))
                elif 'temel analiz' in line:
                    result['fundamental_analysis'] = line.split(':')[1].strip()[:500] if ':' in line else text[:500]
                elif 'yorum' in line:
                    result['comment'] = line.split(':')[1].strip() if ':' in line else text

            # Fiyatlar DeepSeek tarafından sağlanmazsa mevcut fiyat üzerinden yedekleme
            if result['entry_price'] is None:
                result['entry_price'] = current_price
            if result['exit_price'] is None:
                result['exit_price'] = current_price * 1.02
            if result['stop_loss'] is None:
                result['stop_loss'] = current_price * 0.98
            if result['support_level'] is None:
                result['support_level'] = current_price * 0.95
            if result['resistance_level'] is None:
                result['resistance_level'] = current_price * 1.05

            return result
        except Exception as e:
            logger.error(f"DeepSeek yanıtı ayrıştırılırken hata: {e}")
            return {
                'entry_price': current_price,
                'exit_price': current_price * 1.02,
                'stop_loss': current_price * 0.98,
                'leverage': '1x',
                'pump_probability': 50,
                'dump_probability': 50,
                'trend': 'Nötr',
                'support_level': current_price * 0.95,
                'resistance_level': current_price * 1.05,
                'risk_reward_ratio': 1.0,
                'fundamental_analysis': 'Analiz başarısız',
                'comment': self.generate_dynamic_default_comment({'price': current_price})
            }

    def generate_dynamic_default_comment(self, data):
        """Göstergelere dayalı dinamik bir varsayılan yorum üretir."""
        indicators = data.get('indicators', {})
        comment = "Piyasa şu anda belirsiz bir seyir izliyor. "
        for interval in ['1m', '5m', '60m', '1d']:
            rsi = indicators.get(f'rsi_{interval}', 0.0)
            volume_change = indicators.get(f'volume_change_{interval}', 0.0)
            macd = indicators.get(f'macd_{interval}', 0.0)

            if rsi > 70:
                comment += f"{interval}: RSI {rsi:.2f}, aşırı alım bölgesinde, düzeltme riski mevcut. "
            elif rsi < 30:
                comment += f"{interval}: RSI {rsi:.2f}, aşırı satım bölgesinde, alım fırsatı olabilir. "
            else:
                comment += f"{interval}: RSI {rsi:.2f}, nötr bölgede, net bir sinyal yok. "

            if volume_change > 100:
                comment += f"{interval}: Hacimde %{volume_change:.2f} artış, hareketlilik bekleniyor. "
            elif volume_change < -50:
                comment += f"{interval}: Hacimde %{volume_change:.2f} düşüş, piyasa durgun. "
            else:
                comment += f"{interval}: Hacimde %{volume_change:.2f} değişim, dikkatli izlenmeli. "

            if macd > 0:
                comment += f"{interval}: MACD {macd:.2f}, yükseliş eğilimi sinyali. "
            elif macd < 0:
                comment += f"{interval}: MACD {macd:.2f}, düşüş eğilimi sinyali. "

        bid_ask_ratio = indicators.get('bid_ask_ratio', 0.0)
        if bid_ask_ratio > 1.5:
            comment += f"Bid/ask oranı {bid_ask_ratio:.2f}, güçlü alım baskısı mevcut."
        elif bid_ask_ratio < 0.7:
            comment += f"Bid/ask oranı {id_ask_ratio:.2f}, satış baskısı hakim."
        else:
            comment += f"Bid/ask oranı {bid_ask_ratio:.2f}, alım ve satım baskıları dengeli."

        if len(comment) < 500:
            comment += " " * (500 - len(comment))
        return comment[:500]

class Storage:
    """Analiz sonuçlarını ve grup konuşmalarını depolar."""
    def __init__(self):
        self.file_path = "analysis.json"
        self.conversation_file_path = "conversations.json"

    def save_analysis(self, data):
        """Analiz sonuçlarını JSON dosyasına kaydeder."""
        logger.info("Analiz sonuçları kaydediliyor")
        try:
            existing_data = self.load_analysis()
            existing_data.update(data)
            with open(self.file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Analiz kaydedilirken hata: {e}")

    def load_analysis(self):
        """Analiz sonuçlarını JSON dosyasından yükler."""
        logger.info("Analiz sonuçları yükleniyor")
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Analiz yüklenirken hata: {e}")
            return {}

    def save_conversation(self, chat_id, message, timestamp):
        """Grup konuşmalarını kaydeder."""
        logger.info(f"Konuşma kaydediliyor: chat_id={chat_id}")
        try:
            conversations = self.load_conversations()
            if str(chat_id) not in conversations:
                conversations[str(chat_id)] = []
            conversations[str(chat_id)].append({
                'message': message,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
            conversations[str(chat_id)] = conversations[str(chat_id)][-50:]
            with open(self.conversation_file_path, 'w') as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            logger.error(f"Konuşma kaydedilirken hata: {e}")

    def load_conversations(self):
        """Grup konuşmalarını yükler."""
        logger.info("Konuşmalar yükleniyor")
        try:
            with open(self.conversation_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Konuşmalar yüklenirken hata: {e}")
            return {}

def calculate_indicators(kline_data, order_book):
    """Farklı zaman dilimleri için teknik göstergeleri hesaplar."""
    logger.info("Teknik göstergeler hesaplanıyor")
    try:
        indicators = {}
        intervals = ['1m', '5m', '60m', '1d']
        for interval in intervals:
            kline = kline_data.get(interval, [])
            if kline and len(kline) > 1:
                df = pd.DataFrame(
                    kline,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_timestamp', 'quote_volume']
                )
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                volume_change = (df['volume'].iloc[-1] / df['volume'].iloc[-2] - 1) * 100 if df['volume'].iloc[-2] != 0 else 0.0
                indicators[f'volume_change_{interval}'] = volume_change
                rsi = ta.rsi(df['close'], length=14)
                indicators[f'rsi_{interval}'] = rsi.iloc[-1] if not rsi.empty else 0.0
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                indicators[f'macd_{interval}'] = macd['MACD_12_26_9'].iloc[-1] if not macd.empty else 0.0
            else:
                indicators[f'volume_change_{interval}'] = 0.0
                indicators[f'rsi_{interval}'] = 0.0
                indicators[f'macd_{interval}'] = 0.0

        if order_book and 'bids' in order_book and 'asks' in order_book:
            bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
            ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
            indicators['bid_ask_ratio'] = bid_volume / ask_volume if ask_volume > 0 else 0.0
        else:
            indicators['bid_ask_ratio'] = 0.0

        return indicators
    except Exception as e:
        logger.error(f"Göstergeler hesaplanırken hata: {e}")
        return {
            'volume_change_1m': 0.0, 'rsi_1m': 0.0, 'macd_1m': 0.0,
            'volume_change_5m': 0.0, 'rsi_5m': 0.0, 'macd_5m': 0.0,
            'volume_change_60m': 0.0, 'rsi_60m': 0.0, 'macd_60m': 0.0,
            'volume_change_1d': 0.0, 'rsi_1d': 0.0, 'macd_1d': 0.0,
            'bid_ask_ratio': 0.0
        }

def explain_indicators(indicators):
    """Göstergeleri anlaşılır şekilde açıklar."""
    explanations = []
    for interval in ['1m', '5m', '60m', '1d']:
        volume_change = indicators.get(f'volume_change_{interval}', 0.0)
        rsi = indicators.get(f'rsi_{interval}', 0.0)
        macd = indicators.get(f'macd_{interval}', 0.0)

        if isinstance(volume_change, (int, float)):
            if volume_change > 100:
                vol_explain = f"{interval}: Hacimde %{volume_change:.2f} artış, güçlü alım/satım hareketi olabilir."
            elif volume_change > 0:
                vol_explain = f"{interval}: Hacimde %{volume_change:.2f} artış, ilgi artıyor."
            elif volume_change < -50:
                vol_explain = f"{interval}: Hacimde %{volume_change:.2f} düşüş, piyasada sakinlik var."
            else:
                vol_explain = f"{interval}: Hacimde %{volume_change:.2f} değişim, stabil hareket."
        else:
            vol_explain = f"{interval}: Hacim verisi yok."
        explanations.append(vol_explain)

        if isinstance(rsi, (int, float)):
            if rsi > 70:
                rsi_explain = f"{interval}: RSI {rsi:.2f}, aşırı alım bölgesinde, düşüş riski olabilir."
            elif rsi < 30:
                rsi_explain = f"{interval}: RSI {rsi:.2f}, aşırı satım bölgesinde, alım fırsatı olabilir."
            else:
                rsi_explain = f"{interval}: RSI {rsi:.2f}, nötr bölgede, net bir sinyal yok."
        else:
            rsi_explain = f"{interval}: RSI verisi yok."
        explanations.append(rsi_explain)

        if isinstance(macd, (int, float)):
            if macd > 0:
                macd_explain = f"{interval}: MACD {macd:.2f}, yükseliş eğilimi sinyali."
            elif macd < 0:
                macd_explain = f"{interval}: MACD {macd:.2f}, düşüş eğilimi sinyali."
            else:
                macd_explain = f"{interval}: MACD {macd:.2f}, nötr sinyal."
        else:
            macd_explain = f"{interval}: MACD verisi yok."
        explanations.append(macd_explain)

    bid_ask_ratio = indicators.get('bid_ask_ratio', 0.0)
    if isinstance(bid_ask_ratio, (int, float)):
        if bid_ask_ratio > 1.5:
            bid_ask_explain = f"Bid/Ask Oranı: {bid_ask_ratio:.2f}, güçlü alım baskısı var."
        elif bid_ask_ratio < 0.7:
            bid_ask_explain = f"Bid/Ask Oranı: {bid_ask_ratio:.2f}, satış baskısı hakim."
        else:
            bid_ask_explain = f"Bid/Ask Oranı: {bid_ask_ratio:.2f}, dengeli alım/satım."
    else:
        bid_ask_explain = "Bid/Ask Oranı: Veri yok."
    explanations.append(bid_ask_explain)

    return "\n".join(explanations)

class TelegramBot:
    """Telegram botunu başlatır."""
    def __init__(self):
        logger.info("TelegramBot başlatılıyor")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
        self.storage = Storage()
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            logger.error("TELEGRAM_BOT_TOKEN eksik")
            raise ValueError("TELEGRAM_BOT_TOKEN eksik")
        try:
            self.app = Application.builder().token(bot_token).build()
            logger.info("Application başlatıldı")
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("analyze", self.analyze))
            self.app.add_handler(CallbackQueryHandler(self.button))
            self.app.add_handler(CommandHandler("show_analysis", self.show_analysis))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.chat))
            self.web_app = None
            self.active_analyses = {}
            if self.app.job_queue:
                self.app.job_queue.start()
                logger.info("Job queue başlatıldı")
        except Exception as e:
            logger.error(f"Application başlatılırken hata: {e}")
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start komutu için menü gösterir."""
        logger.info(f"Start komutu alındı, chat_id={update.effective_chat.id}")
        keyboard = [
            [InlineKeyboardButton("Top 10 Spot Analizi", callback_data='top_10_spot')],
            [InlineKeyboardButton("Top 100 Spot Analizi", callback_data='top_100_spot')],
            [InlineKeyboardButton("Top 10 Vadeli Analizi", callback_data='top_10_futures')],
            [InlineKeyboardButton("Top 100 Vadeli Analizi", callback_data='top_100_futures')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Analiz için butonları kullanabilir veya /analyze <symbol> [trade_type] komutuyla coin analizi yapabilirsiniz (örn. /analyze BTCUSDT spot).",
            reply_markup=reply_markup
        )

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Belirtilen sembol için analiz yapar."""
        logger.info(f"Analyze komutu alındı: {update.message.text}")
        try:
            args = update.message.text.split()
            if len(args) < 2:
                await update.message.reply_text("Lütfen bir sembol girin. Örnek: /analyze BTCUSDT spot")
                return
            symbol = args[1].upper()
            trade_type = args[2].lower() if len(args) > 2 else 'spot'
            if trade_type not in ['spot', 'futures']:
                await update.message.reply_text("Trade tipi 'spot' veya 'futures' olmalı. Örnek: /analyze BTCUSDT spot")
                return
            if not symbol.endswith('USDT'):
                await update.message.reply_text("Sembol USDT çifti olmalı. Örnek: /analyze BTCUSDT")
                return

            analysis_key = f"{symbol}_{trade_type}_{update.effective_chat.id}"
            if analysis_key in self.active_analyses:
                await update.message.reply_text(f"{symbol} için {trade_type} analizi zaten yapılıyor, lütfen bekleyin.")
                return
            self.active_analyses[analysis_key] = True

            mexc = MEXCClient()
            if not await mexc.validate_symbol(symbol):
                del self.active_analyses[analysis_key]
                await update.message.reply_text(f"Hata: {symbol} geçersiz bir işlem çifti.")
                return

            await update.message.reply_text(f"{symbol} için {trade_type} analizi yapılıyor...")
            data = await self.process_coin(symbol, mexc, trade_type, update.effective_chat.id)
            await mexc.close()
            del self.active_analyses[analysis_key]

            if not data:
                await update.message.reply_text(f"{symbol} için analiz yapılamadı.")
        except Exception as e:
            logger.error(f"Analyze komutunda hata: {e}")
            await update.message.reply_text(f"Hata: {str(e)}")
            if analysis_key in self.active_analyses:
                del self.active_analyses[analysis_key]

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton tıklamalarını işler."""
        query = update.callback_query
        await query.answer()
        logger.info(f"Buton tıklandı: {query.data}")
        try:
            parts = query.data.split('_')
            limit = int(parts[1])
            trade_type = parts[2]
            await query.message.reply_text(f"{trade_type.upper()} analizi yapılıyor (Top {limit})...")
            data = {'chat_id': self.group_id, 'limit': limit, 'trade_type': trade_type}
            if self.app.job_queue is None:
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
        """Top coin'ler için analiz yapar ve gönderir."""
        if data is None:
            data = context.job.data
        chat_id = data['chat_id']
        limit = data['limit']
        trade_type = data['trade_type']
        logger.info(f"Analiz yapılıyor: chat_id={chat_id}, limit={limit}, trade_type={trade_type}")
        try:
            if limit == 100 and trade_type == 'spot':
                results = await self.analyze_top_100_from_endpoints(chat_id, trade_type)
            else:
                results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"Top {limit} {trade_type} analizi için sonuç bulunamadı.")
            logger.info(f"Top {limit} {trade_type} için analiz tamamlandı")
        except Exception as e:
            logger.error(f"analyze_and_send sırasında hata: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"{trade_type} analizi sırasında hata: {str(e)}")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Genel mesajlara yanıt verir ve konuşmaları kaydeder."""
        message_text = update.message.text
        logger.info(f"Mesaj alındı: {message_text}")
        try:
            self.storage.save_conversation(
                chat_id=update.effective_chat.id,
                message=message_text,
                timestamp=datetime.now()
            )
            deepseek = DeepSeekClient()
            conversations = self.storage.load_conversations()
            group_context = conversations.get(str(update.effective_chat.id), [])
            context_str = "\n".join([f"[{c['timestamp']}] {c['message']}" for c in group_context])

            prompt = f"""
            Aşağıdaki grup konuşma mesajına Türkçe, doğal ve bağlama uygun bir yanıt ver. Yanıt minimum 500 karakter, maksimum 5000 karakter olmalı. Grup konuşma geçmişini dikkate alarak mesajı kişiselleştir ve eğer mesaj bir soru içeriyorsa (örneğin, 'yükselecek mi?'), bu soruya doğrudan hitap et. Sabit veya tekrarlayan ifadelerden kaçın, akıcı ve profesyonel bir üslup kullan. Eğer mesaj bir coin'le ilgiliyse (örneğin, 'BTCUSDT yükselecek mi?'), genel piyasa bilgisi veya son trendlere dayanarak kısa bir analiz yap.

            Mesaj: {message_text}
            Grup konuşma geçmişi:
            {context_str if context_str else 'Grup konuşma geçmişi yok.'}
            """
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    deepseek.client.chat.completions.create,
                    model="deepseek-moe",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5000
                ),
                timeout=20
            )
            response_text = response.choices[0].message.content
            if len(response_text) < 500:
                response_text += " " * (500 - len(response_text))
            await update.message.reply_text(response_text)
        except Exception as e:
            logger.error(f"Chat sırasında hata: {e}")
            await update.message.reply_text(f"Hata: {str(e)}")

    async def show_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Kaydedilen analizleri gösterir."""
        logger.info("show_analysis komutu alındı")
        try:
            data = self.storage.load_analysis()
            messages = []
            for key, coin_data in data.items():
                if isinstance(coin_data, dict) and 'coin' in coin_data:
                    trade_type = key.split('_')[-1]
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
        """Analiz sonuçlarını formatlar."""
        logger.info(f"{symbol} ({trade_type}) için sonuçlar biçimlendiriliyor")
        indicators = coin_data.get('indicators', {})
        analysis = coin_data.get('deepseek_analysis', {}).get('short_term', {})
        try:
            message = (
                f" {symbol} {trade_type.upper()} Analizi ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                f"- Kısa Vadeli:\n"
                f"  - Giriş: ${analysis.get('entry_price', 'Belirtilmemiş')}\n"
                f"  - Çıkış: ${analysis.get('exit_price', 'Belirtilmemiş')}\n"
                f"  - Stop Loss: ${analysis.get('stop_loss', 'Belirtilmemiş')}\n"
                f"  - Kaldıraç: {analysis.get('leverage', 'Bilinmiyor')}\n"
                f"- Trend: {analysis.get('trend', 'Nötr')}\n"
                f"- Pump Olasılığı: {analysis.get('pump_probability', 0)}%\n"
                f"- Dump Olasılığı: {analysis.get('dump_probability', 0)}%\n"
                f"- Destek Seviyesi: ${analysis.get('support_level', 'Belirtilmemiş')}\n"
                f"- Direnç Seviyesi: ${analysis.get('resistance_level', 'Belirtilmemiş')}\n"
                f"- Risk/Ödül Oranı: {analysis.get('risk_reward_ratio', 0):.2f}\n"
                f"- Temel Analiz: {analysis.get('fundamental_analysis', 'Veri yok')}\n"
                f"- Göstergeler:\n"
            )
            for interval in ['1m', '5m', '60m', '1d']:
                message += (
                    f"  - Hacim Değişimi: {interval}: {indicators.get(f'volume_change_{interval}', 'Bilinmiyor') if isinstance(indicators.get(f'volume_change_{interval}'), (int, float)) else 'Bilinmiyor'}%\n"
                    f"  - RSI: {interval}: {indicators.get(f'rsi_{interval}', 'Bilinmiyor') if isinstance(indicators.get(f'rsi_{interval}'), (int, float)) else 'Bilinmiyor'}\n"
                    f"  - MACD: {interval}: {indicators.get(f'macd_{interval}', 'Bilinmiyor') if isinstance(indicators.get(f'macd_{interval}'), (int, float)) else 'Bilinmiyor'}\n"
                )
            message += (
                f"  - Bid/Ask Oranı: {indicators.get('bid_ask_ratio', 'Bilinmiyor') if isinstance(indicators.get('bid_ask_ratio'), (int, float)) else 'Bilinmiyor'}\n"
                f"- Gösterge Açıklamaları:\n{explain_indicators(indicators)}\n"
                f"- DeepSeek Yorumu: {analysis.get('comment', 'Yorum yok.')}"
            )
            return message
        except Exception as e:
            logger.error(f"{symbol} için sonuçlar biçimlendirilirken hata: {e}")
            return f"{symbol} analizi biçimlendirilirken hata: {str(e)}"

    async def process_coin(self, symbol, mexc, trade_type, chat_id, endpoint=None):
        """Tek bir coin için analiz yapar."""
        logger.info(f"{symbol} ({trade_type}) işleniyor")
        try:
            data = await mexc.fetch_and_save_market_data(symbol, endpoint)
            if not data or not any(data.get('klines', {}).get(interval) for interval in ['1m', '5m', '60m', '1d']):
                logger.warning(f"{symbol} ({trade_type}) için geçerli piyasa verisi yok")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için geçerli piyasa verisi yok")
                return None

            data['indicators'] = calculate_indicators(
                data['klines'],
                data.get('order_book')
            )
            if not data['indicators']:
                logger.warning(f"{symbol} ({trade_type}) için gösterge hesaplanamadı")
                await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} için gösterge hesaplanamadı")
                return None

            deepseek = DeepSeekClient()
            data['deepseek_analysis'] = deepseek.analyze_coin(symbol, data, trade_type, chat_id)
            data['coin'] = symbol

            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"{symbol} ({trade_type}) için analiz gönderildi")

            self.storage.save_analysis({f'{symbol}_{trade_type}': data})
            return data
        except Exception as e:
            logger.error(f"{symbol} ({trade_type}) işlenirken hata: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"{symbol} işlenirken hata: {str(e)}")
            return None

    async def analyze_coins(self, limit, trade_type, chat_id):
        """Top coin'ler için analiz yapar."""
        logger.info(f"analyze_coins başlatılıyor: limit={limit}, trade_type={trade_type}")
        mexc = MEXCClient()
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_{limit}_{trade_type}': []}
        coins = await mexc.get_top_coins(limit)
        logger.info(f"{len(coins)} coin analiz ediliyor: {coins[:5]}...")

        for symbol in coins:
            analysis_key = f"{symbol}_{trade_type}_{chat_id}"
            if analysis_key in self.active_analyses:
                logger.info(f"{symbol} için analiz zaten yapılıyor, atlanıyor")
                continue
            self.active_analyses[analysis_key] = True
            coin_data = await self.process_coin(symbol, mexc, trade_type, chat_id)
            if coin_data:
                results[f'top_{limit}_{trade_type}'].append(coin_data)
            del self.active_analyses[analysis_key]
            await asyncio.sleep(1)
        await mexc.close()
        return results

    async def analyze_top_100_from_endpoints(self, chat_id, trade_type):
        """GitHub'daki endpoints.json'dan top 100 coin için analiz yapar."""
        logger.info(f"Top 100 {trade_type} analizi endpoints.json'dan yapılıyor")
        mexc = MEXCClient()
        results = {'date': datetime.now().strftime('%Y-%m-%d'), f'top_100_{trade_type}': []}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ENDPOINTS_JSON_URL) as response:
                    if response.status != 200:
                        logger.error(f"endpoints.json alınamadı: {response.status}")
                        await self.app.bot.send_message(chat_id=chat_id, text="endpoints.json alınamadı.")
                        return results
                    endpoints = await response.json()

            for entry in endpoints:
                symbol = entry['symbol']
                endpoint = entry['endpoint']
                analysis_key = f"{symbol}_{trade_type}_{chat_id}"
                if analysis_key in self.active_analyses:
                    logger.info(f"{symbol} için analiz zaten yapılıyor, atlanıyor")
                    continue
                self.active_analyses[analysis_key] = True
                coin_data = await self.process_coin(symbol, mexc, trade_type, chat_id, endpoint)
                if coin_data:
                    results[f'top_100_{trade_type}'].append(coin_data)
                del self.active_analyses[analysis_key]
                await asyncio.sleep(1)
            await mexc.close()
            return results
        except Exception as e:
            logger.error(f"Top 100 analizi sırasında hata: {e}")
            await self.app.bot.send_message(chat_id=chat_id, text=f"Top 100 analizi sırasında hata: {str(e)}")
            return results

    async def webhook_handler(self, request):
        """Webhook isteklerini işler."""
        logger.info("Webhook isteği alındı")
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
        logger.info("Webhook sunucusu başlatılıyor")
        await self.app.initialize()
        await self.app.start()
        self.web_app = web.Application()
        self.web_app.router.add_post('/webhook', self.webhook_handler)
        webhook_url = f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com/webhook"
        try:
            await self.app.bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook {webhook_url} adresine ayarlandı")
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
    logger.info("Main başlatılıyor")
    bot = TelegramBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    logger.info("Script başlatıldı")
    main()

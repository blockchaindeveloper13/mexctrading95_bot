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

class MEXCClient:
    """MEXC API ile iletişim kurar."""
    def __init__(self):
        self.base_url = "https://api.mexc.com"

    async def fetch_and_save_market_data(self, symbol):
        """Belirtilen sembol için piyasa verisi çeker."""
        logger.info(f"{symbol} için piyasa verisi çekiliyor")
        try:
            async with aiohttp.ClientSession() as session:
                klines = {}
                for timeframe in ['1m', '5m', '15m', '30m', '60m']:
                    url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={timeframe}&limit=100"
                    async with session.get(url) as response:
                        if response.status == 200:
                            klines[timeframe] = await response.json()
                        else:
                            logger.warning(f"{symbol} için {timeframe} kline verisi alınamadı: {response.status}")
                            klines[timeframe] = []
                    await asyncio.sleep(1)

                order_book_url = f"{self.base_url}/api/v3/depth?symbol={symbol}&limit=10"
                async with session.get(order_book_url) as order_book_response:
                    order_book = await order_book_response.json() if order_book_response.status == 200 else {}
                await asyncio.sleep(1)

                ticker_url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(ticker_url) as ticker_response:
                    ticker = await ticker_response.json() if ticker_response.status == 200 else {'price': '0.0'}
                await asyncio.sleep(1)

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

    def parse_deepseek_response(self, text, current_price):
        """DeepSeek yanıtını parse eder."""
        try:
            entry_price = float(re.search(r'Entry Price: (\d+\.?\d*)', text).group(1)) if re.search(r'Entry Price: (\d+\.?\d*)', text) else current_price
            exit_price = float(re.search(r'Exit Price: (\d+\.?\d*)', text).group(1)) if re.search(r'Exit Price: (\d+\.?\d*)', text) else current_price * 1.02
            stop_loss = float(re.search(r'Stop Loss: (\d+\.?\d*)', text).group(1)) if re.search(r'Stop Loss: (\d+\.?\d*)', text) else current_price * 0.98
            leverage = re.search(r'Leverage: ([0-9x]+)', text).group(1) if re.search(r'Leverage: ([0-9x]+)', text) else '1x'
            pump_prob = int(re.search(r'Pump Probability: (\d+)%', text).group(1)) if re.search(r'Pump Probability: (\d+)%', text) else 50
            dump_prob = int(re.search(r'Dump Probability: (\d+)%', text).group(1)) if re.search(r'Dump Probability: (\d+)%', text) else 50
            trend = re.search(r'Trend: (\w+)', text).group(1) if re.search(r'Trend: (\w+)', text) else 'Neutral'
            support = float(re.search(r'Support Level: (\d+\.?\d*)', text).group(1)) if re.search(r'Support Level: (\d+\.?\d*)', text) else current_price * 0.95
            resistance = float(re.search(r'Resistance Level: (\d+\.?\d*)', text).group(1)) if re.search(r'Resistance Level: (\d+\.?\d*)', text) else current_price * 1.05
            risk_reward = float(re.search(r'Risk/Reward Ratio: (\d+\.?\d*)', text).group(1)) if re.search(r'Risk/Reward Ratio: (\d+\.?\d*)', text) else (exit_price - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 1.0
            fundamental = re.search(r'Fundamental Analysis: (.+?)(?:\n|$)', text).group(1)[:500] if re.search(r'Fundamental Analysis: (.+?)(?:\n|$)', text) else text[:500]
            comment = re.search(r'Comment: (.+?)(?:\n|$)', text).group(1)[:500] if re.search(r'Comment: (.+?)(?:\n|$)', text) else "Hold: Insufficient signals for strong buy or sell. Monitor for volume increase."
            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'leverage': leverage,
                'pump_probability': pump_prob,
                'dump_probability': dump_prob,
                'trend': trend,
                'support_level': support,
                'resistance_level': resistance,
                'risk_reward_ratio': risk_reward,
                'fundamental_analysis': fundamental,
                'comment': comment
            }
        except Exception as e:
            logger.error(f"DeepSeek yanıtı parse edilirken hata: {e}")
            return {
                'entry_price': current_price,
                'exit_price': current_price * 1.02,
                'stop_loss': current_price * 0.98,
                'leverage': '1x',
                'pump_probability': 50,
                'dump_probability': 50,
                'trend': 'Neutral',
                'support_level': current_price * 0.95,
                'resistance_level': current_price * 1.05,
                'risk_reward_ratio': 1.0,
                'fundamental_analysis': 'Parse başarısız',
                'comment': 'Hold: Insufficient signals for strong buy or sell. Monitor for volume increase.'
            }

    def analyze_coin(self, symbol, data, trade_type):
        """DeepSeek API ile coin analizi yapar."""
        logger.info(f"{symbol} için {trade_type} analizi yapılıyor")
        try:
            prompt = f"""
            Analyze {symbol} for {trade_type} trading strategy:
            - Current Price: {data['price']} USDT
            - Volume Changes: 
              - 1m: {data.get('indicators', {}).get('volume_change_1m', 'N/A')}%
              - 5m: {data.get('indicators', {}).get('volume_change_5m', 'N/A')}%
              - 15m: {data.get('indicators', {}).get('volume_change_15m', 'N/A')}%
              - 30m: {data.get('indicators', {}).get('volume_change_30m', 'N/A')}%
              - 60m: {data.get('indicators', {}).get('volume_change_60m', 'N/A')}%
            - RSI: 
              - 1m: {data.get('indicators', {}).get('rsi_1m', 'N/A')}
              - 5m: {data.get('indicators', {}).get('rsi_5m', 'N/A')}
              - 15m: {data.get('indicators', {}).get('rsi_15m', 'N/A')}
              - 30m: {data.get('indicators', {}).get('rsi_30m', 'N/A')}
              - 60m: {data.get('indicators', {}).get('rsi_60m', 'N/A')}
            - MACD: 
              - 1m: {data.get('indicators', {}).get('macd_1m', 'N/A')}
              - 5m: {data.get('indicators', {}).get('macd_5m', 'N/A')}
              - 15m: {data.get('indicators', {}).get('macd_15m', 'N/A')}
              - 30m: {data.get('indicators', {}).get('macd_30m', 'N/A')}
              - 60m: {data.get('indicators', {}).get('macd_60m', 'N/A')}
            - Bid/Ask Ratio: {data.get('indicators', {}).get('bid_ask_ratio', 'N/A')}
            Provide:
            - Entry Price: <specific price in USDT>
            - Exit Price: <specific price in USDT>
            - Stop Loss: <specific price in USDT>
            - Leverage: <e.g., 1x for spot, 3x for futures>
            - Pump Probability: <%> (based on RSI, volume, and bid/ask ratio)
            - Dump Probability: <%> (based on RSI, volume, and bid/ask ratio)
            - Trend: <Bullish/Bearish/Neutral>
            - Support Level: <specific price in USDT>
            - Resistance Level: <specific price in USDT>
            - Risk/Reward Ratio: <e.g., 2.0>
            - Fundamental Analysis: <detailed summary, max 500 characters, include volume trends, market sentiment, and buying/selling pressure>
            - Comment: <specific trading recommendation (Buy/Sell/Hold) with detailed reasoning based on indicators, max 500 characters>
            """
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            analysis_text = response.choices[0].message.content
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
                    'trend': 'Neutral',
                    'support_level': data['price'] * 0.95,
                    'resistance_level': data['price'] * 1.05,
                    'risk_reward_ratio': 1.0,
                    'fundamental_analysis': 'Analiz başarısız',
                    'comment': 'Hold: Insufficient signals for strong buy or sell. Monitor for volume increase.'
                }
            }

class Storage:
    """Analiz sonuçlarını depolar ve yükler."""
    def __init__(self):
        self.file_path = "analysis.json"

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

def calculate_indicators(kline_1m, kline_5m, kline_15m, kline_30m, kline_60m, order_book):
    """Teknik göstergeleri hesaplar."""
    logger.info("Teknik göstergeler hesaplanıyor")
    try:
        indicators = {}
        for timeframe, kline in [('1m', kline_1m), ('5m', kline_5m), ('15m', kline_15m), ('30m', kline_30m), ('60m', kline_60m)]:
            if kline and len(kline) > 1:
                df = pd.DataFrame(
                    kline,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_timestamp', 'quote_volume']
                )
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                volume_change = (df['volume'].iloc[-1] / df['volume'].iloc[-2] - 1) * 100 if df['volume'].iloc[-2] != 0 else 0.0
                indicators[f'volume_change_{timeframe}'] = volume_change
                rsi = ta.rsi(df['close'], length=14)
                indicators[f'rsi_{timeframe}'] = rsi.iloc[-1] if not rsi.empty else 0.0
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                indicators[f'macd_{timeframe}'] = macd['MACD_12_26_9'].iloc[-1] if not macd.empty else 0.0
            else:
                indicators[f'volume_change_{timeframe}'] = 0.0
                indicators[f'rsi_{timeframe}'] = 0.0
                indicators[f'macd_{timeframe}'] = 0.0

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
            'volume_change_15m': 0.0, 'rsi_15m': 0.0, 'macd_15m': 0.0,
            'volume_change_30m': 0.0, 'rsi_30m': 0.0, 'macd_30m': 0.0,
            'volume_change_60m': 0.0, 'rsi_60m': 0.0, 'macd_60m': 0.0,
            'bid_ask_ratio': 0.0
        }

def explain_indicators(indicators):
    """Göstergeleri anlaşılır şekilde açıklar."""
    explanations = []
    for tf in ['1m', '5m', '15m', '30m', '60m']:
        volume_change = indicators.get(f'volume_change_{tf}', 0.0)
        rsi = indicators.get(f'rsi_{tf}', 0.0)
        macd = indicators.get(f'macd_{tf}', 0.0)
        
        if isinstance(volume_change, (int, float)):
            if volume_change > 100:
                vol_explain = f"{tf}: Hacimde %{volume_change:.2f} artış, güçlü alım/satım hareketi olabilir."
            elif volume_change > 0:
                vol_explain = f"{tf}: Hacimde %{volume_change:.2f} artış, ilgi artıyor."
            elif volume_change < -50:
                vol_explain = f"{tf}: Hacimde %{volume_change:.2f} düşüş, piyasada sakinlik var."
            else:
                vol_explain = f"{tf}: Hacimde %{volume_change:.2f} değişim, stabil hareket."
        else:
            vol_explain = f"{tf}: Hacim verisi yok."
        explanations.append(vol_explain)

        if isinstance(rsi, (int, float)):
            if rsi > 70:
                rsi_explain = f"{tf}: RSI {rsi:.2f}, aşırı alım bölgesinde, düşüş riski olabilir."
            elif rsi < 30:
                rsi_explain = f"{tf}: RSI {rsi:.2f}, aşırı satım bölgesinde, alım fırsatı olabilir."
            else:
                rsi_explain = f"{tf}: RSI {rsi:.2f}, nötr bölgede, net bir sinyal yok."
        else:
            rsi_explain = f"{tf}: RSI verisi yok."
        explanations.append(rsi_explain)

        if isinstance(macd, (int, float)):
            if macd > 0:
                macd_explain = f"{tf}: MACD {macd:.2f}, yükseliş eğilimi sinyali."
            elif macd < 0:
                macd_explain = f"{tf}: MACD {macd:.2f}, düşüş eğilimi sinyali."
            else:
                macd_explain = f"{tf}: MACD {macd:.2f}, nötr sinyal."
        else:
            macd_explain = f"{tf}: MACD verisi yok."
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
    def __init__(self):
        """Telegram botunu başlatır."""
        logger.info("TelegramBot başlatılıyor")
        self.group_id = int(os.getenv('TELEGRAM_GROUP_ID', '-1002869335730'))
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
            self.active_analyses = {}  # Aynı sembol için tekrar eden analizleri önlemek
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
            results = await self.analyze_coins(limit, trade_type, chat_id)
            if not results.get(f'top_{limit}_{trade_type}'):
                await context.bot.send_message(chat_id=chat_id, text=f"Top {limit} {trade_type} analizi için sonuç bulunamadı.")
            logger.info(f"Top {limit} {trade_type} için analiz tamamlandı")
        except Exception as e:
            logger.error(f"analyze_and_send sırasında hata: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"{trade_type} analizi sırasında hata: {str(e)}")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Genel mesajlara yanıt verir."""
        logger.info(f"Mesaj alındı: {update.message.text}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.app.bot.client.chat.completions.create,
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
        """Kaydedilen analizleri gösterir."""
        logger.info("show_analysis komutu alındı")
        try:
            storage = Storage()
            data = storage.load_analysis()
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
        volume_changes = {}
        for tf in ['1m', '5m', '15m', '30m', '60m']:
            value = indicators.get(f'volume_change_{tf}', 0.0)
            volume_changes[tf] = f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"
        bid_ask_ratio = indicators.get('bid_ask_ratio', 0.0)
        bid_ask_ratio_str = f"{bid_ask_ratio:.2f}" if isinstance(bid_ask_ratio, (int, float)) else "N/A"
        rsi_values = {tf: indicators.get(f'rsi_{tf}', 0.0) for tf in ['1m', '5m', '15m', '30m', '60m']}
        macd_values = {tf: indicators.get(f'macd_{tf}', 0.0) for tf in ['1m', '5m', '15m', '30m', '60m']}
        try:
            message = (
                f"📊 {symbol} {trade_type.upper()} Analizi ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
                f"- Kısa Vadeli:\n"
                f"  - Giriş: ${analysis.get('entry_price', 0):.2f}\n"
                f"  - Çıkış: ${analysis.get('exit_price', 0):.2f}\n"
                f"  - Stop Loss: ${analysis.get('stop_loss', 0):.2f}\n"
                f"  - Kaldıraç: {analysis.get('leverage', 'N/A')}\n"
                f"- Trend: {analysis.get('trend', 'Neutral')}\n"
                f"- Pump Olasılığı: {analysis.get('pump_probability', 0)}%\n"
                f"- Dump Olasılığı: {analysis.get('dump_probability', 0)}%\n"
                f"- Destek Seviyesi: ${analysis.get('support_level', 0):.2f}\n"
                f"- Direnç Seviyesi: ${analysis.get('resistance_level', 0):.2f}\n"
                f"- Risk/Ödül Oranı: {analysis.get('risk_reward_ratio', 0):.2f}\n"
                f"- Temel Analiz: {analysis.get('fundamental_analysis', 'Veri yok')}\n"
                f"- Göstergeler:\n"
                f"  - Hacim Değişimleri: 1m: {volume_changes['1m']}% | 5m: {volume_changes['5m']}% | "
                f"15m: {volume_changes['15m']}% | 30m: {volume_changes['30m']}% | 60m: {volume_changes['60m']}%\n"
                f"  - Bid/Ask Oranı: {bid_ask_ratio_str}\n"
                f"  - RSI: 1m: {rsi_values['1m']:.2f} | 5m: {rsi_values['5m']:.2f} | "
                f"15m: {rsi_values['15m']:.2f} | 30m: {rsi_values['30m']:.2f} | 60m: {rsi_values['60m']:.2f}\n"
                f"  - MACD: 1m: {macd_values['1m']:.2f} | 5m: {macd_values['5m']:.2f} | "
                f"15m: {macd_values['15m']:.2f} | 30m: {macd_values['30m']:.2f} | 60m: {macd_values['60m']:.2f}\n"
                f"- Gösterge Açıklamaları:\n{explain_indicators(indicators)}\n"
                f"- DeepSeek Yorumu: {analysis.get('comment', 'Yorum yok.')}"
            )
            return message
        except Exception as e:
            logger.error(f"{symbol} için sonuçlar biçimlendirilirken hata: {e}")
            return f"{symbol} analizi biçimlendirilirken hata: {str(e)}"

    async def process_coin(self, symbol, mexc, trade_type, chat_id):
        """Tek bir coin için analiz yapar."""
        logger.info(f"{symbol} ({trade_type}) işleniyor")
        try:
            data = await mexc.fetch_and_save_market_data(symbol)
            if not data or not data.get('klines', {}).get('1m'):
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

            deepseek = DeepSeekClient()
            data['deepseek_analysis'] = deepseek.analyze_coin(symbol, data, trade_type)
            data['coin'] = symbol

            message = self.format_results(data, trade_type, symbol)
            await self.app.bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"{symbol} ({trade_type}) için analiz gönderildi")

            storage = Storage()
            storage.save_analysis({f'{symbol}_{trade_type}': data})

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
            await asyncio.sleep(2)
        await mexc.close()
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

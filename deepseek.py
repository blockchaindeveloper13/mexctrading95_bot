import os
import json
import logging
import re
from openai import OpenAI
from dotenv import load_dotenv

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def clean_json_response(self, response):
        """DeepSeek yanıtından markdown ve ek metinleri temizler, sadece JSON kısmını döndürür."""
        try:
            cleaned = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE)
            cleaned = re.sub(r'\n\s*###.*|\n\s*\*.*', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}, original response: {response}")
            return response

    def analyze_coin(self, data, trade_type='spot'):
        try:
            symbol = data.get('coin', 'Unknown')
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            indicators = data.get('indicators', {})
            
            prompt = (
                f"Analyze the following cryptocurrency data for {symbol} ({trade_type.upper()} trading):\n"
                f"Current Price: ${price}\n"
                f"24h Volume: ${volume}\n"
                f"1h RSI: {indicators.get('rsi_1h', 'N/A')}\n"
                f"1h EMA20: {indicators.get('ema_20_1h', 'N/A')}\n"
                f"1h EMA50: {indicators.get('ema_50_1h', 'N/A')}\n"
                f"1h MACD: {indicators.get('macd_1h', 'N/A')}\n"
                f"1h MACD Signal: {indicators.get('macd_signal_1h', 'N/A')}\n"
                f"1h Bollinger Upper: {indicators.get('bb_upper_1h', 'N/A')}\n"
                f"1h Bollinger Lower: {indicators.get('bb_lower_1h', 'N/A')}\n"
                f"1h VWAP: {indicators.get('vwap_1h', 'N/A')}\n"
                f"1h Stochastic %K: {indicators.get('stoch_k_1h', 'N/A')}\n"
                f"1h Stochastic %D: {indicators.get('stoch_d_1h', 'N/A')}\n"
                f"1h ATR: {indicators.get('atr_1h', 'N/A')}\n"
                f"4h RSI: {indicators.get('rsi_4h', 'N/A')}\n"
                f"4h EMA20: {indicators.get('ema_20_4h', 'N/A')}\n"
                f"4h EMA50: {indicators.get('ema_50_4h', 'N/A')}\n"
                f"4h MACD: {indicators.get('macd_4h', 'N/A')}\n"
                f"4h MACD Signal: {indicators.get('macd_signal_4h', 'N/A')}\n"
                f"4h Bollinger Upper: {indicators.get('bb_upper_4h', 'N/A')}\n"
                f"4h Bollinger Lower: {indicators.get('bb_lower_4h', 'N/A')}\n"
                f"4h VWAP: {indicators.get('vwap_4h', 'N/A')}\n"
                f"4h Stochastic %K: {indicators.get('stoch_k_4h', 'N/A')}\n"
                f"4h Stochastic %D: {indicators.get('stoch_d_4h', 'N/A')}\n"
                f"4h ATR: {indicators.get('atr_4h', 'N/A')}\n"
                f"Volume Change 1h: {indicators.get('volume_change_1h', 'N/A')}%;\n"
                f"Volume Change 3h: {indicators.get('volume_change_3h', 'N/A')}%;\n"
                f"Volume Change 6h: {indicators.get('volume_change_6h', 'N/A')}%;\n"
                f"Volume Change 24h: {indicators.get('volume_change_24h', 'N/A')}%;\n"
                f"Bid/Ask Ratio: {indicators.get('bid_ask_ratio', 'N/A')}\n"
                "Perform a comprehensive short-term trading analysis (1-4 hours) including technical and fundamental factors. For fundamental analysis, include recent news or market sentiment if available (use your deep search capabilities). Provide:\n"
                "- pump_probability (0-100%)\n"
                "- dump_probability (0-100%)\n"
                "- entry_price\n"
                "- exit_price\n"
                "- stop_loss\n"
                "- leverage (string, e.g., '2x', '5x', or 'Not recommended' for {trade_type} trading)\n"
                "- fundamental_analysis (string, brief summary of news/market sentiment)\n"
                "Return *only* a valid JSON object with these fields, no additional text, markdown, or comments."
            )
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = response.choices[0].message.content
            cleaned_analysis = self.clean_json_response(analysis)
            try:
                parsed = json.loads(cleaned_analysis)
                if not isinstance(parsed, dict):
                    logger.warning(f"DeepSeek response for {symbol} ({trade_type}) is not a valid JSON object: {cleaned_analysis}")
                    return {'short_term': {
                        'pump_probability': 0,
                        'dump_probability': 0,
                        'entry_price': price,
                        'exit_price': price,
                        'stop_loss': price * 0.95,
                        'leverage': 'N/A',
                        'fundamental_analysis': 'No data available'
                    }}
                required_keys = ['pump_probability', 'dump_probability', 'entry_price', 'exit_price', 'stop_loss', 'leverage', 'fundamental_analysis']
                if not all(key in parsed for key in required_keys):
                    logger.warning(f"DeepSeek response for {symbol} ({trade_type}) missing required keys: {cleaned_analysis}")
                    return {'short_term': {
                        'pump_probability': 0,
                        'dump_probability': 0,
                        'entry_price': price,
                        'exit_price': price,
                        'stop_loss': price * 0.95,
                        'leverage': 'N/A',
                        'fundamental_analysis': 'No data available'
                    }}
                logger.info(f"DeepSeek analysis completed for {symbol} ({trade_type})")
                return {'short_term': parsed}
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek response for {symbol} ({trade_type}) is not valid JSON: {cleaned_analysis}, error: {e}")
                return {'short_term': {
                    'pump_probability': 0,
                    'dump_probability': 0,
                    'entry_price': price,
                    'exit_price': price,
                    'stop_loss': price * 0.95,
                    'leverage': 'N/A',
                    'fundamental_analysis': 'No data available'
                }}
        except Exception as e:
            logger.error(f"Error analyzing coin {symbol} ({trade_type}): {e}")
            return {'short_term': {
                'pump_probability': 0,
                'dump_probability': 0,
                'entry_price': price,
                'exit_price': price,
                'stop_loss': price * 0.95,
                'leverage': 'N/A',
                'fundamental_analysis': 'No data available'
            }}

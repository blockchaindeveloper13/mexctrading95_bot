import os
import json
import logging
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

    def analyze_coin(self, data):
        try:
            symbol = data.get('coin', 'Unknown')  # Use 'coin' instead of 'symbol'
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            indicators = data.get('indicators', {})
            
            prompt = (
                f"Analyze the following cryptocurrency data for {symbol}:\n"
                f"Current Price: ${price}\n"
                f"24h Volume: ${volume}\n"
                f"1h RSI: {indicators.get('rsi_1h', 'N/A')}\n"
                f"1h EMA20: {indicators.get('ema_20_1h', 'N/A')}\n"
                f"1h EMA50: {indicators.get('ema_50_1h', 'N/A')}\n"
                f"1h MACD: {indicators.get('macd_1h', 'N/A')}\n"
                f"1h MACD Signal: {indicators.get('macd_signal_1h', 'N/A')}\n"
                f"4h RSI: {indicators.get('rsi_4h', 'N/A')}\n"
                f"4h EMA20: {indicators.get('ema_20_4h', 'N/A')}\n"
                f"4h EMA50: {indicators.get('ema_50_4h', 'N/A')}\n"
                f"4h MACD: {indicators.get('macd_4h', 'N/A')}\n"
                f"4h MACD Signal: {indicators.get('macd_signal_4h', 'N/A')}\n"
                "Provide a short-term trading analysis (1-4 hours) including:\n"
                "- Pump probability (0-100%)\n"
                "- Dump probability (0-100%)\n"
                "- Entry price\n"
                "- Exit price\n"
                "- Stop loss\n"
                "- Leverage (if applicable)\n"
                "Return the response in JSON format."
            )
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = response.choices[0].message.content
            try:
                # Try to parse JSON response
                parsed = json.loads(analysis)
                if not isinstance(parsed, dict):
                    logger.warning(f"DeepSeek response for {symbol} is not a valid JSON object: {analysis}")
                    return {'short_term': {
                        'pump_probability': 0,
                        'dump_probability': 0,
                        'entry_price': price,
                        'exit_price': price,
                        'stop_loss': price * 0.95,
                        'leverage': 'N/A'
                    }}
                logger.info(f"DeepSeek analysis completed for {symbol}")
                return {'short_term': parsed}
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek response for {symbol} is not valid JSON: {analysis}, error: {e}")
                return {'short_term': {
                    'pump_probability': 0,
                    'dump_probability': 0,
                    'entry_price': price,
                    'exit_price': price,
                    'stop_loss': price * 0.95,
                    'leverage': 'N/A'
                }}
        except Exception as e:
            logger.error(f"Error analyzing coin {symbol}: {e}")
            return {'short_term': {
                'pump_probability': 0,
                'dump_probability': 0,
                'entry_price': data.get('price', 0),
                'exit_price': data.get('price', 0),
                'stop_loss': data.get('price', 0) * 0.95,
                'leverage': 'N/A'
            }}

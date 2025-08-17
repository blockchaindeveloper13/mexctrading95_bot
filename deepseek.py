from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def analyze_coin(self, data):
        try:
            prompt = f"""
            Aşağıdaki verilere göre coin’in kısa ve uzun vadeli fiyat hareketlerini analiz et:
            - Coin: {data['symbol']}
            - 24h Hacim Değişimi: {data['volume_change']}%
            - Fiyat Değişimi: {data['price_change']}%
            - Alış/Satış Oranı: {data['bid_ask_ratio']}
            - Kısa Vadeli (1h): RSI: {data['indicators']['short_term'].get('rsi', 0)}, Stochastic: {data['indicators']['short_term'].get('stoch_k', 0)}, MA 50: {data['indicators']['short_term'].get('ma_50', 0)}, EMA 12: {data['indicators']['short_term'].get('ema_12', 0)}, MACD: {data['indicators']['short_term'].get('macd', 0)}, Bollinger: {data['indicators']['short_term'].get('upper_band', 0)}, KDJ: {data['indicators']['short_term'].get('kdj', 0)}, SAR: {data['indicators']['short_term'].get('sar', 0)}, AVL: {data['indicators']['short_term'].get('avl', 0)}, Volume: {data['indicators']['short_term'].get('volume', 0)}, OBV: {data['indicators']['short_term'].get('obv', 0)}, WR: {data['indicators']['short_term'].get('wr', 0)}
            - Uzun Vadeli (4h): RSI: {data['indicators']['long_term'].get('rsi', 0)}, Stochastic: {data['indicators']['long_term'].get('stoch_k', 0)}, MA 50: {data['indicators']['long_term'].get('ma_50', 0)}, EMA 12: {data['indicators']['long_term'].get('ema_12', 0)}, MACD: {data['indicators']['long_term'].get('macd', 0)}, Bollinger: {data['indicators']['long_term'].get('upper_band', 0)}, KDJ: {data['indicators']['long_term'].get('kdj', 0)}, SAR: {data['indicators']['long_term'].get('sar', 0)}, AVL: {data['indicators']['long_term'].get('avl', 0)}, Volume: {data['indicators']['long_term'].get('volume', 0)}, OBV: {data['indicators']['long_term'].get('obv', 0)}, WR: {data['indicators']['long_term'].get('wr', 0)}
            JSON formatında cevap ver:
            {{
              "coin": "{data['symbol']}",
              "short_term": {{
                "entry_price": {{price}},
                "exit_price": {{price}},
                "stop_loss": {{price}},
                "leverage": "{{leverage}}x",
                "pump_probability": {{percentage}},
                "dump_probability": {{percentage}},
                "analysis": "{{text}}"
              }},
              "long_term": {{
                "entry_price": {{price}},
                "exit_price": {{price}},
                "stop_loss": {{price}},
                "leverage": "{{leverage}}x",
                "pump_probability": {{percentage}},
                "dump_probability": {{percentage}},
                "analysis": "{{text}}"
              }}
            }}
            Pump/dump olasılıklarını yüzde olarak belirt, teknik terimler kullan, riskleri vurgula, kaldıraç oranını öner.
            """
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                response_format={'type': 'json_object'},
                max_tokens=4096
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing coin {data['symbol']}: {e}")
            return {
                "coin": data['symbol'],
                "short_term": {"analysis": "Error in analysis"},
                "long_term": {"analysis": "Error in analysis"}
            }

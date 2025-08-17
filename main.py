from mexc_api import MEXCClient
from indicators import calculate_indicators
from deepseek import DeepSeekClient
from telegram_bot import TelegramBot
from storage import Storage
from datetime import datetime
import asyncio

def analyze_coins(limit):
    mexc = MEXCClient()
    deepseek = DeepSeekClient()
    storage = Storage()
    
    coins = mexc.get_top_coins(limit)
    results = {'date': datetime.now().strftime('%Y-%m-%d'), 'top_100': [], 'top_300': []}
    
    for symbol in coins:
        data = mexc.get_market_data(symbol)
        if data:
            data['indicators'] = calculate_indicators(data['klines_1h'], data['klines_4h'])
            data['deepseek_analysis'] = deepseek.analyze_coin(data)
            results['top_100' if limit == 100 else 'top_300'].append(data)
    
    storage.save_analysis(results)
    return results

def main():
    bot = TelegramBot(analyze_coins)
    bot.run()

if __name__ == "__main__":
    main()

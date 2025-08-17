import json
import logging
import os

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Storage:
    def save_analysis(self, data):
        try:
            if not data or not any(data.get(key, []) for key in data if key.startswith('top_')):
                logger.warning("No valid data to save in analysis.json")
                return
            with open('analysis.json', 'w') as f:
                json.dump(data, f, indent=4)
            logger.info("Analysis saved to analysis.json")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    def load_analysis(self):
        try:
            with open('analysis.json', 'r') as f:
                data = json.load(f)
            logger.info("Analysis loaded from analysis.json")
            return data
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return {}

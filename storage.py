import json
import logging
import os

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Storage:
    def save_analysis(self, data):
        logger.debug(f"Saving analysis data: {data}")
        try:
            if not data or not any(data.get(key, []) for key in data if key.startswith('top_')):
                logger.warning("No valid data to save in analysis.json")
                return
            with open('/tmp/analysis.json', 'w') as f:  # Heroku için /tmp
                json.dump(data, f, indent=4)
            logger.info("Analysis saved to /tmp/analysis.json")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    def load_analysis(self):
        logger.debug("Loading analysis from /tmp/analysis.json")
        try:
            with open('/tmp/analysis.json', 'r') as f:
                data = json.load(f)
            logger.info("Analysis loaded from /tmp/analysis.json")
            logger.debug(f"Loaded analysis data: {data}")
            return data
        except FileNotFoundError:
            logger.warning("Analysis file not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            return {}

import json
from datetime import datetime

class Storage:
    def save_analysis(self, data):
        try:
            with open('analysis.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving analysis: {e}")

    def load_analysis(self):
        try:
            with open('analysis.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

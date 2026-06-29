import requests
import os
from dotenv import load_dotenv
import json

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

url = "https://api.adzuna.com/v1/api/jobs/in/search/1"

params = {
    "app_id": APP_ID,
    "app_key": APP_KEY,
    "results_per_page": 5,
    "what": "python developer",
    "where": "bangalore"
}

response = requests.get(url, params=params)
data = response.json()

print(json.dumps(data, indent=2))
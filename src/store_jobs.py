import requests
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
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
jobs = data["results"]

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

for job in jobs:
    cursor.execute(
        """
        INSERT INTO jobs (id, title, description, company, location, created, redirect_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            job["id"],
            job["title"],
            job["description"],
            job["company"]["display_name"],
            job["location"]["display_name"],
            job["created"],
            job["redirect_url"]
        )
    )

conn.commit()
cursor.close()
conn.close()

print(f"Inserted {len(jobs)} jobs into the database.")

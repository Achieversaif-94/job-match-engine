import os
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

def fetch_jobs(query, location, results=5):
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_APP_KEY}&results_per_page={results}&what={query}&where={location}"
    response = requests.get(url)
    data = response.json()
    return data.get("results", [])

def store_jobs(jobs):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    for job in jobs:
        cur.execute(
            "INSERT INTO jobs (id, title, description, company, location, redirect_url) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (job["id"], job["title"], job["description"], job.get("company", {}).get("display_name", "Unknown"), job.get("location", {}).get("display_name", "Unknown"), job.get("redirect_url", ""))
        )
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "python developer"
    location = sys.argv[2] if len(sys.argv) > 2 else "bangalore"
    jobs = fetch_jobs(query, location)
    store_jobs(jobs)
    print(f"Fetched and stored {len(jobs)} jobs for '{query}' in {location}")
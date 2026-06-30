import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        company TEXT,
        location TEXT,
        created TIMESTAMP,
        redirect_url TEXT
    )
""")

conn.commit()
cur.close()
conn.close()

print("Table created successfully")
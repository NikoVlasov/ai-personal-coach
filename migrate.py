import os
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing in .env")

# Parse the URL for psycopg2
url = urlparse(DATABASE_URL)

conn = psycopg2.connect(
    host=url.hostname,
    port=url.port,
    dbname=url.path[1:],
    user=url.username,
    password=url.password,
    sslmode="require"
)

cur = conn.cursor()

migrations = [
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS age INTEGER",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS limitations TEXT",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS days_per_week INTEGER",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
    """
    CREATE TABLE IF NOT EXISTS workout_logs (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        exercise VARCHAR,
        sets INTEGER,
        reps INTEGER,
        duration_minutes INTEGER,
        notes TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    )
    """,
]

for sql in migrations:
    try:
        cur.execute(sql)
        print(f"OK: {sql.strip()[:60]}...")
    except Exception as e:
        print(f"SKIP: {e}")

conn.commit()
cur.close()
conn.close()
print("\nMigration complete!")
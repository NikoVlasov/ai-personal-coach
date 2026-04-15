import os
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing in .env")

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
    """
    CREATE TABLE IF NOT EXISTS workout_logs (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        created_at TIMESTAMP DEFAULT NOW()
    )
    """,
    "ALTER TABLE workout_logs ADD COLUMN IF NOT EXISTS exercise VARCHAR",
    "ALTER TABLE workout_logs ADD COLUMN IF NOT EXISTS sets INTEGER",
    "ALTER TABLE workout_logs ADD COLUMN IF NOT EXISTS reps INTEGER",
    "ALTER TABLE workout_logs ADD COLUMN IF NOT EXISTS duration_minutes INTEGER",
    "ALTER TABLE workout_logs ADD COLUMN IF NOT EXISTS notes TEXT",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS age INTEGER",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS limitations TEXT",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS days_per_week INTEGER",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
    "ALTER TABLE fitness_profiles ALTER COLUMN weight TYPE FLOAT USING weight::float",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS language VARCHAR DEFAULT 'en'",
    "ALTER TABLE fitness_profiles ADD COLUMN IF NOT EXISTS schedule VARCHAR DEFAULT ''",
    """
    CREATE TABLE IF NOT EXISTS daily_checkins (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        energy INTEGER,
        soreness INTEGER,
        mood INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    )
    """,
]

print("Starting migration...\n")

for sql in migrations:
    try:
        cur.execute(sql)
        conn.commit()
        print(f"OK: {sql.strip()[:70]}...")
    except Exception as e:
        conn.rollback()
        print(f"SKIP ({e.__class__.__name__}): {sql.strip()[:50]}...")

print("\n--- workout_logs columns ---")
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'workout_logs'
    ORDER BY ordinal_position
""")
for row in cur.fetchall():
    print(f"  {row[0]} -- {row[1]}")

print("\n--- fitness_profiles columns ---")
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'fitness_profiles'
    ORDER BY ordinal_position
""")
for row in cur.fetchall():
    print(f"  {row[0]} -- {row[1]}")

cur.close()
conn.close()
print("\nMigration complete!")
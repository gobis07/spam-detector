import sqlite3

conn = sqlite3.connect("message.db")
cursor = conn.cursor()

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS message(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    prediction TEXT,
    created_at TIMESTAMP DEFAULT
CURRENT_TIMESTAMP
    )
    """
)
conn.commit()
conn.close()
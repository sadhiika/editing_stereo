from pathlib import Path
import sqlite3


def init_schema(db_path: str = "stereowipe.db", schema_path: str = "database/schema.sql") -> None:
    db_file = Path(db_path)
    schema_file = Path(schema_path)

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    conn = sqlite3.connect(db_file)
    try:
        conn.executescript(schema_file.read_text())
        conn.commit()
    finally:
        conn.close()

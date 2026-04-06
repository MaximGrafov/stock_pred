# create_db.py
# Скрипт создает оболочку БД, для заполнения данными

import sqlite3

from pathlib import Path

from support_functions import load_config, require_config_value


def create_db() -> None:

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)
    
    database_file = require_config_value(config, 'paths.database_file')
    
    database_path = project_root / database_file
    database_path.parent.mkdir(parents=True, exist_ok=True)


    with sqlite3.connect(database_path) as connect:
        
        cursor = connect.cursor()


        cursor.executescript("""--sql
                       CREATE TABLE IF NOT EXISTS stock_prices
                       (
                           id     INTEGER PRIMARY KEY AUTOINCREMENT,
                           date   TEXT NOT NULL,
                           ticker TEXT NOT NULL,
                           open   REAL NOT NULL,
                           high   REAL NOT NULL,
                           low    REAL NOT NULL,
                           close  REAL NOT NULL,
                           volume REAL NOT NULL,
                           UNIQUE(date, ticker)
                       );


                       CREATE INDEX IF NOT EXISTS idx_stock_ticker_data
                       ON stock_prices(ticker, date);

                       """)
        
    print(f'База подготовлена: {database_path}')

    
if __name__ == '__main__':
    create_db()



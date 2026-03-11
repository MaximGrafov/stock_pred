# create_db.py
# Скрипт создает оболочку БД, для заполнения данными

import sqlite3

connect = sqlite3.connect('main_db.db')


def create_db():

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

    connect.commit()
    connect.close()

    
if __name__ == '__main__':
    create_db()



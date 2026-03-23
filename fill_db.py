# fill_db.py
# Скрипт заполняет БД

import time
import sqlite3
import pandas as pd
import yfinance as yf

from pathlib import Path
from tqdm import tqdm

from additional_functions import (
    load_config, require_config_value,
    choice_tickers
)

project_root = Path(__file__).resolve().parent
config = load_config(project_root)





tickers = choice_tickers()
if not isinstance(tickers, list) or not tickers:
    raise ValueError('tickers.all_tickers должен быть не пустым списком')


start_date = require_config_value(config, 'market_data.start_date')
end_date = require_config_value(config, 'market_data.end_date')
data_interval = require_config_value(config, 'market_data.data_interval')
request_sleep_seconds = require_config_value(config, 'market_data.request_sleep_seconds')


database_file = require_config_value(config, 'paths.database_file')
database_path = project_root / database_file
database_path.parent.mkdir(parents=True, exist_ok=True)


def load_one_ticker(ticker: str) -> pd.DataFrame:
    
    from typing import cast
    
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=data_interval,
        auto_adjust=False,
        progress=False)
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = cast(pd.DataFrame, df).copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()

    df.columns = [str(s).lower() for s in df.columns]

    df['ticker'] = ticker
    
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].round(2)
    df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna()
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
    return df


def fill_db() -> None:
     
    insert_sql = """--sql
                    INSERT OR IGNORE INTO stock_prices
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    
    total_inserted = 0
    
    with sqlite3.connect(database_path) as connect:
        cursor = connect.cursor()
    
        for ticker in tqdm(tickers, desc='Загрузка тикеров', unit='тик'):

                time.sleep(request_sleep_seconds)
                df = load_one_ticker(ticker)

                if df.empty:
                    print(f'[ОШИБКА] Для тикера {ticker} нет данных')
                    continue

                rows = [
                    (
                        row['date'],
                        row['ticker'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    )
                    for _, row in df.iterrows()
                    ]

                before = connect.total_changes
                cursor.executemany(insert_sql, rows)
                connect.commit()
                inserted_now = connect.total_changes - before
                total_inserted += inserted_now

                print(f'\n{ticker}: было добавлено {inserted_now} строк')
    
    
        cursor.execute('SELECT COUNT(*) FROM stock_prices')
        total_rows = cursor.fetchone()[0]
        
    print(f"""
          Итог:
          Было добавлено {total_inserted} строк
          Всего в stock_prices {total_rows} строк
          """)
    
    
if __name__ == '__main__':
    fill_db()
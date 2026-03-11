# fill_db.py
# Скрипт заполняет БД

import time
import sqlite3
import pandas as pd
import yfinance as yf

from tqdm import tqdm

connect = sqlite3.connect('main_db.db')

TICKERS = [
    # Tech / Internet
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ADBE", "CRM", "ORCL",
    "AMD", "INTC", "CSCO", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC",
    "SNPS", "CDNS", "PANW", "CRWD", "NET", "SHOP", "UBER", "ABNB", "PYPL", "SQ",

    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "BK",

    # Healthcare / Pharma
    "JNJ", "PFE", "MRK", "LLY", "ABBV", "BMY", "TMO", "DHR", "MDT", "ABT",

    # Consumer
    "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "PEP", "KO",
    "PG", "CL", "KMB", "EL", "UL",

    # Energy / Industrial / Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "DE", "BA", "GE", "HON",
    "MMM", "LIN", "APD", "NEM", "FCX",

    # Telecom / Utilities / REIT
    "VZ", "T", "TMUS", "NEE", "DUK", "SO", "D", "PLD", "AMT", "CCI",

    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "VTI", "XLF", "XLK", "XLV", "XLE", "XLI",
    "XLP", "XLY", "XLB", "XLU", "VNQ", "SMH", "ARKK", "TLT", "GLD", "USO"
]

START_DATE = '2015-01-01'
END_DATE = '2026-03-12'


def load_one_ticker(ticker: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=START_DATE,
        end=END_DATE,
        interval='1d',
        auto_adjust=False,
        progress=False)
    
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()

    df.columns = [str(s).lower() for s in df.columns]


    df['ticker'] = ticker
    
    df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna()
    
    
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
    return df


def fill_db() -> None:
    
    cursor = connect.cursor()
    
    insert_sql = """--sql
                    INSERT OR IGNORE INTO stock_prices
                    (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    
    total_inserted = 0
    
    for ticker in tqdm(TICKERS, desc='Загрузка тикеров', unit='тик'):
        
        time.sleep(1)
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

        print(f'{ticker}: было добавлено {inserted_now} строк')
    
    
    cursor.execute('SELECT COUNT(*) FROM stock_prices')
    total_rows = cursor.fetchone()[0]
    
    connect.close()
    
    print(f"""
          Итог:
          Было добавлено {total_inserted} строк
          Всего в stock_prices {total_rows} строк
          """)
    
    
if __name__ == '__main__':
    fill_db()
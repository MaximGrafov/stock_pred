# fill_db.py
# Скрипт заполняет БД

import time
import sqlite3
import pandas as pd
import yfinance as yf

from tqdm import tqdm

connect = sqlite3.connect('main_db.db')

TICKERS = [
    "AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP",
    "ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB",
    "ALGN","ALK","ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP",
    "AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD","APH","APTV",
    "ARE","ATO","AVB","AVGO","AVY","AWK","AXP","AZO","BA","BAC",
    "BALL","BAX","BBWI","BBY","BDX","BEN","BG","BIIB","BK","BKNG",
    "BKR","BLK","BMY","BR","BRO","BSX","BWA","BX","BXP","C",
    "CAG","CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDNS",
    "CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF",
    "CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COO","COP","COST","CPB","CPRT","CPT","CRL","CRM","CSCO",
    "CSGP","CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR",
    "D","DAL","DD","DE","DFS","DG","DGX","DHI","DHR","DIS",
    "DLR","DLTR","DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN",
    "DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV",
    "EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES","ESS",
    "ETN","ETR","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG",
    "FAST","FCX","FDS","FDX","FE","FFIV","FICO","FIS","FISV","FITB",
    "FLT","FMC","FOX","FOXA","FRT","FTNT","FTV","GD","GE","GEHC",
    "GEN","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
    "GPN","GRMN","GS","HAL","HAS","HBAN","HCA","HD","HES","HIG",
    "HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY",
    "HUM","HWM","IBM","ICE","IDXX","IEX","IFF","ILMN","INCY","INTC",
    "INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW",
    "J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP",
    "KEY","KEYS","KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR",
    "LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW",
    "LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR",
    "MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM",
    "MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOS","MPC",
    "MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD",
    "MU","NCLH","NDAQ","NEE","NEM","NFLX","NI","NKE","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI",
    "O","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS","PARA","PAYC",
    "PAYX","PCAR","PCG","PEAK","PEG","PEP","PFE","PFG","PG","PGR",
    "PH","PHM","PKG","PLD","PM","PNC","PNR","PNW","PODD","POOL",
    "PPG","PPL","PRU","PSA","PSX","PTC","PWR","PYPL","QCOM","QRVO",
    "RCL","REG","REGN","RF","RHI","RJF","RL","RMD","ROK","ROL",
    "ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW"
    ]

START_DATE = '2010-01-01'
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
    
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].round(2)
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

        print(f'\n{ticker}: было добавлено {inserted_now} строк')
    
    
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
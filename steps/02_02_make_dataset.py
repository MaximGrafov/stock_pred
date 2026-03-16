# 02_mnake_dataset.py
#

import sqlite3
import pandas as pd

from pathlib import Path

connect = sqlite3.connect('./main_db.db')

root = Path(__file__).resolve().parents[1]
db_path = root /'main_db.db'

data_dir = root /'data'
data_dir.mkdir(exist_ok=True)


df = pd.read_sql_query("""--sql
                        SELECT date, ticker, open, high, low, close, volume
                            FROM stock_prices
                        ORDER BY ticker, date
                       """, connect)

connect.close()


df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date']).reset_index(drop=True) # type: pd.DataFrame


group = df.groupby('ticker', group_keys=False)

# Признаки доходности

df['returns_1']  = group['close'].pct_change(1)
df['returns_3']  = group['close'].pct_change(3)
df['returns_5']  = group['close'].pct_change(5)
df['returns_10'] = group['close'].pct_change(10)


# Признаки тренда

df['moving_avg_5'] = group['close'].transform(lambda x: x.rolling(5).mean())
df['moving_avg_10'] = group['close'].transform(lambda x: x.rolling(10).mean())
df['moving_avg_20'] = group['close'].transform(lambda x: x.rolling(20).mean())


# Волатильность и относительность

df['volume_10'] = group['returns_1'].transform(lambda x: x.rolling(10).std())
df['close_mavg_5_ratio'] = df['close'] / df['moving_avg_5']
df['close_mavg_20_ratio'] = df['close'] / df['moving_avg_20']


future_close = group['close'].shift(-5)
df['target'] = ((future_close / df['close'] - 1.0) > 0).astype(int)


feature_cols = ['returns_1', 'returns_3', 'returns_5', 'returns_10',
                'moving_avg_5', 'moving_avg_10', 'moving_avg_20',
                'volume_10', 'close_mavg_5_ratio', 'close_mavg_20_ratio']

df = df.dropna(subset=feature_cols + ['target']).copy()


out_parquet = data_dir / 'dataset.parquet'
df.to_parquet(out_parquet, index=False)


print(f"""
      Программа выполнена!
      Всего строк: {len(df)}
      Всего колонок: {len(df.columns)}
      Доля target=1: {round(df['target'].mean(), 4)}
      Parquet: {out_parquet}
      """)
# 02_mnake_dataset.py
#

import sqlite3
import pandas as pd

from pathlib import Path



root = Path(__file__).resolve().parents[1]
db_path = root /'main_db.db'

data_dir = root /'data'
data_dir.mkdir(exist_ok=True)

connect = sqlite3.connect(db_path)


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

for col in ['returns_1', 'returns_3', 'returns_5', 'returns_10']:
    df[col] = df[col].clip(-0.2, 0.2)


# Признаки тренда

df['moving_avg_5'] = group['close'].transform(lambda x: x.rolling(5).mean())
df['moving_avg_10'] = group['close'].transform(lambda x: x.rolling(10).mean())
df['moving_avg_20'] = group['close'].transform(lambda x: x.rolling(20).mean())


# Доп. признаки цены

df['day_range'] = (df['high'] - df['low']) / df['close']
df['day_body'] = (df['close'] - df['open']) / df['open']


# Объемные признаки

df['volume_change_1'] = group['volume'].pct_change(1)
df['volume_moving_avg_20'] = group['volume'].transform(lambda x: x.rolling(20).mean())
df['volume_ratio_20'] = df['volume'] / df['volume_moving_avg_20']


# Нормирование (z-score) доходности внутри тикера

ret_mean_20 = group['returns_1'].transform(lambda x: x.rolling(20).mean())
ret_std_20 = group['returns_1'].transform(lambda x: x.rolling(20).std())
df['ret1_z_20'] = (df['returns_1'] - ret_mean_20) / (ret_std_20 + 1e-9)


# Волатильность и относительность

df['volume_10'] = group['returns_1'].transform(lambda x: x.rolling(10).std())
df['close_mavg_5_ratio'] = df['close'] / df['moving_avg_5']
df['close_mavg_20_ratio'] = df['close'] / df['moving_avg_20']


future_close = group['close'].shift(-5)
df['future_ret_5'] = future_close / df['close'] - 1.0


eps = 0.01
df = df[df['future_ret_5'].abs() >= eps].copy()

df['target'] = (df['future_ret_5'] > 0).astype(int)


feature_columns = ['returns_1', 'returns_3', 'returns_5', 'returns_10',
                'moving_avg_5', 'moving_avg_10', 'moving_avg_20',
                'volume_10', 'close_mavg_5_ratio', 'close_mavg_20_ratio',
                'day_range', 'day_body', 'volume_change_1',
                'volume_moving_avg_20', 'volume_ratio_20', 'ret1_z_20'
                  ]


df = df.dropna(subset=feature_columns + ['target', 'future_ret_5']).copy()
df = df.drop(columns=['future_ret_5'])

out_parquet = data_dir / 'dataset.parquet'
df.to_parquet(out_parquet, index=False)


print(f"""
      Программа выполнена!
      Всего строк: {len(df)}
      Всего колонок: {len(df.columns)}
      Доля target=1: {round(df['target'].mean(), 4)}
      Parquet: {out_parquet}
      """)
# 02_mnake_dataset.py
#

import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path

from additional_functions import load_config, require_config_value


project_root = Path(__file__).resolve().parents[1]
config = load_config(project_root)

database_file = require_config_value(config, 'paths.database_file')
dataset_parquet = require_config_value(config, 'paths.dataset_parquet')

returns_windows = require_config_value(config, 'dataset.returns_windows')
moving_average_windows = require_config_value(config, 'dataset.moving_average_windows')
returns_clip_range = require_config_value(config, 'dataset.returns_clip_range')
target_horizon_days = require_config_value(config, 'dataset.target_horizon_days')
target_absolute_minimum_return = require_config_value(config, 'dataset.target_absolute_minimum_return')
_features = require_config_value(config, 'features.dataset_columns')

database_path = project_root / database_file

with sqlite3.connect(database_path) as connect:


    df = pd.read_sql_query("""--sql
                            SELECT date, ticker, open, high, low, close, volume
                                FROM stock_prices
                            ORDER BY ticker, date
                           """, connect)



    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True) # type: pd.DataFrame


    group = df.groupby('ticker', group_keys=False)

    # Признаки доходности

    df['returns_1']  = group['close'].pct_change(1)
    df['returns_3']  = group['close'].pct_change(3)
    df['returns_5']  = group['close'].pct_change(5)
    df['returns_10'] = group['close'].pct_change(10)

    for col in ['returns_1', 'returns_3', 'returns_5', 'returns_10']:
        df[col] = df[col].clip(returns_clip_range[0], returns_clip_range[1])


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


    future_close = group['close'].shift(-target_horizon_days)
    df['future_ret_5'] = future_close / df['close'] - 1.0

    df = df[df['future_ret_5'].abs() >= float(target_absolute_minimum_return)].copy()

    df['target'] = (df['future_ret_5'] > 0).astype(int)


    numeric_columns, feature_columns= _features.copy(), _features.copy()    

    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=feature_columns + ['target', 'future_ret_5']).copy()
    df = df.drop(columns=['future_ret_5'])

    out_parquet = project_root / dataset_parquet
    df.to_parquet(out_parquet, index=False)


print(f"""
      Программа выполнена!
      Всего строк: {len(df)}
      Всего колонок: {len(df.columns)}
      Доля target=1: {round(df['target'].mean(), 4)}
      Parquet: {out_parquet}
      """)
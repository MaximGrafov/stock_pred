# 04_baseline_hgb.py
# 

from math import isinf

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score
    )
from additional_functions import print_eval_block, write_metrics_reports


root = Path(__file__).resolve().parents[1]
data_path = root / 'data' / 'dataset.parquet'
reports_dir = root / 'reports'
reports_dir.mkdir(exist_ok=True)


df = pd.read_parquet(data_path)
df['date'] = pd.to_datetime(df['date'])


feature_columns = ['ticker', 'returns_1', 'returns_3', 'returns_5',
                   'returns_10', 'moving_avg_5', 'moving_avg_10',
                   'moving_avg_20', 'volume_10', 'close_mavg_5_ratio',
                   'close_mavg_20_ratio', 'day_range', 'day_body',
                   'volume_change_1', 'volume_moving_avg_20',
                   'volume_ratio_20', 'ret1_z_20'
                  ]

for column in feature_columns + ['target']:
    if column not in df.columns:
        raise ValueError(f'Колонка {column} отсутствует')
    
numeric_columns = [column for column in feature_columns if column != 'ticker']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

df[numeric_columns] = df[numeric_columns].clip(-1e6, 1e6)
df = df.dropna(subset=numeric_columns + ['target']).copy()


    
df = df.sort_values('date').reset_index(drop=True)
unique_dates = df['date'].unique()
n_dates = len(unique_dates)


cut1 = int(n_dates * 0.70)
cut2 = int(n_dates * 0.85)


train_end = unique_dates[cut1 - 1]
val_end = unique_dates[cut2 - 1]


train_df = df[df['date'] <= train_end].copy()
val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
test_df = df[df['date'] > val_end].copy()


x_train, y_train = train_df[feature_columns], train_df['target']
x_val, y_val = val_df[feature_columns], val_df['target']
x_test, y_test = test_df[feature_columns], test_df['target']


numeric_features = [column for column in feature_columns if column != 'ticker']
categorical_features = ['ticker']

numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median'))
          ]
)


categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ], sparse_threshold=0.0
)


model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    min_samples_leaf=100,
    random_state=42
)


clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)


if np.isinf(x_train[numeric_features].to_numpy()).any():
    raise ValueError('В x_train остались inf после очистки')

if np.isnan(x_train[numeric_features].to_numpy()).any():
    raise ValueError('В x_train остались NaN после очистки')

clf.fit(x_train, y_train)


val_proba = clf.predict_proba(x_val)[:, 1]
best_threshold = 0.50
best_bal_acc = 0.0


for thr in np.arange(0.30, 0.71, 0.01):
    pred = (val_proba >= thr).astype(int)
    bal_acc = balanced_accuracy_score(y_val, pred)
    if bal_acc > best_bal_acc:
        best_bal_acc = float(bal_acc)
        best_threshold = float(thr)

print(f'Best threshold on VAL: {best_threshold:.2f}, balanced_accuary: {best_bal_acc:.4f}')


def evalute(name: str, x_part: pd.DataFrame, y_part: pd.Series, threshold: float = 0.5):
    proba = clf.predict_proba(x_part)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    
    acc = accuracy_score(y_part, pred)
    bal_acc = balanced_accuracy_score(y_part, pred)
    f1 = f1_score(y_part, pred)
    auc = roc_auc_score(y_part, proba)
    positive_rate = pred.mean()
    
    
    print_eval_block(
        name=name,
        threshold=threshold,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        f1=f1,
        auc_roc=auc,
        positive_rate=positive_rate,
        y_part=y_part,
        pred=pred
        )
    
    return acc, bal_acc, f1, auc
    

val_metrics = evalute('Validation', x_val, y_val, best_threshold)
test_metrics = evalute('Test', x_test, y_test, best_threshold)

val_metrics_map = {
    'accuracy': float(val_metrics[0]),
    'balanced_accuracy': float(val_metrics[1]),
    'f1': float(val_metrics[2]),
    'auc_roc': float(val_metrics[3])
    }

test_metrics_map = {
    'accuracy': float(test_metrics[0]),
    'balanced_accuracy': float(test_metrics[1]),
    'f1': float(test_metrics[2]),
    'auc_roc': float(test_metrics[3])
    }

report_path = reports_dir / 'hgb_metrics.txt'
write_metrics_reports(
    report_path=report_path,
    model_name='Baseline HighGradientBoosting',
    train_rows=len(x_train),
    val_rows=len(x_val),
    test_rows=len(x_test),
    val_metrics=val_metrics_map,
    test_metrics=test_metrics_map
    )
# 03_baseline.py
# 

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,roc_auc_score,\
    balanced_accuracy_score
    )
from additional_functions import print_eval_block, write_metrics_reports



root = Path(__file__).resolve().parents[1]
data_path = root / 'data' / 'dataset.parquet'
reports_dir = root / 'reports'
reports_dir.mkdir(exist_ok=True)


df = pd.read_parquet(data_path)
df['date'] = pd.to_datetime(df['date'])

if 'target' not in df.columns:
    raise ValueError('Отсутствует колонка target в датасете')


feature_columns = [
                    'ticker', 'returns_1', 'returns_3', 'returns_5', 'returns_10',
                    'moving_avg_5', 'moving_avg_10', 'moving_avg_20',
                    'volume_10', 'close_mavg_5_ratio', 'close_mavg_20_ratio'
                  ]


for column in feature_columns:
    if column not in df.columns:
        raise ValueError(f'Нет колонки с именем {column}')


df = df.sort_values('date').reset_index(drop=True)
unique_dates = df['date'].sort_values().unique()

n_dates = len(unique_dates)

cut1 = int(n_dates * 0.70)
cut2 = int(n_dates * 0.85)

train_end_date = unique_dates[cut1 - 1]
val_end_date = unique_dates[cut2 - 1]


train_df = df[df['date'] <= train_end_date].copy()
val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
test_df = df[df['date'] > val_end_date].copy()

x_train, y_train = train_df[feature_columns], train_df['target']
x_val, y_val     = val_df[feature_columns], val_df['target']
x_test, y_test   = test_df[feature_columns], test_df['target']
    

numeric_features = [
                    'returns_1', 'returns_3', 'returns_5', 'returns_10',
                    'moving_avg_5', 'moving_avg_10', 'moving_avg_20',
                    'volume_10', 'close_mavg_5_ratio', 'close_mavg_20_ratio'
                   ]

categorical_features = ['ticker']


numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
          ]
                              )


categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
          ]
                                  )


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
                 ]
                                )


clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'))
          ]
              )


clf.fit(x_train, y_train)

val_proba = clf.predict_proba(x_val)[:, 1]

best_threshold = 0.50
best_bal_acc = 0.0

for thr in np.arange(0.35, 0.66, 0.01):
    val_pred = (val_proba >= thr).astype(int)
    bal_acc = balanced_accuracy_score(y_val, val_pred)
    
    if bal_acc > best_bal_acc:
        best_bal_acc = float(bal_acc)
        best_threshold = float(thr)
        
print(f'Best threshold on VAL: {best_threshold:.2f}, accuarcy: {best_bal_acc:.4f}')


def evaluate(name: str, x_part: pd.DataFrame, y_part: pd.Series, threshold: float = 0.5):
    
    proba = clf.predict_proba(x_part)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    acc = accuracy_score(y_part, pred)
    bal_acc = balanced_accuracy_score(y_part, pred)
    f1  = f1_score(y_part, pred)
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
    

val_metrics = evaluate('Validation', x_val, y_val, threshold=best_threshold)
test_metrics = evaluate('Test', x_test, y_test, threshold=best_threshold)

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


report_path = reports_dir / 'baseline_metrics.txt'
write_metrics_reports(
    report_path=report_path,
    model_name='Baseline Logistic Regression',
    train_rows=len(x_train),
    val_rows=len(x_val),
    test_rows=len(x_test),
    val_metrics=val_metrics_map,
    test_metrics=test_metrics_map
    )

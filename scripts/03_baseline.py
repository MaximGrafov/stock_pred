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
from additional_functions import (
    print_eval_block, write_metrics_reports,
    load_config, require_config_value, show_info
    )


project_root = Path(__file__).resolve().parents[1]
config = load_config(project_root)


dataset_file = project_root / Path(require_config_value(config, 'paths.dataset_parquet'))
report_file = project_root / Path(require_config_value(config, 'paths.baseline_report'))


features = require_config_value(config, 'features')
model_values = require_config_value(config, 'models.logistic_regression')
threshold_parametrs = require_config_value(config, 'threshold_search')
values_threshold = threshold_parametrs['logistic_regression_baseline']

split_values = require_config_value(config, 'split')

training_ratio, validation_ratio = float(split_values['training_ratio']), float(split_values['validation_ratio'])
baseline_features = features['baseline']
categorical_feature_value = features['categorical']
validation_end_ratio = training_ratio + validation_ratio



df = pd.read_parquet(dataset_file)
df['date'] = pd.to_datetime(df['date'])

if 'target' not in df.columns:
    raise ValueError('Отсутствует колонка target в датасете')


feature_columns = baseline_features.copy()

for column in feature_columns:
    if column not in df.columns:
        raise ValueError(f'Нет колонки с именем {column}')


df = df.sort_values('date').reset_index(drop=True)
unique_dates = df['date'].sort_values().unique()

n_dates = len(unique_dates)

cut1 = int(n_dates * training_ratio)
cut2 = int(n_dates * validation_end_ratio)

train_end_date = unique_dates[cut1 - 1]
val_end_date = unique_dates[cut2 - 1]


train_df = df[df['date'] <= train_end_date].copy()
val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
test_df = df[df['date'] > val_end_date].copy()

x_train, y_train = train_df[feature_columns], train_df['target']
x_val, y_val     = val_df[feature_columns], val_df['target']
x_test, y_test   = test_df[feature_columns], test_df['target']
    

numeric_features = [col for col in feature_columns if col != 'ticker']

categorical_features = categorical_feature_value

report_tickers = sorted(train_df['ticker'].dropna().astype(str).unique().tolist())


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
        ('model', LogisticRegression(max_iter=model_values['maximum_iterations'],
                                     n_jobs=model_values['number_of_jobs'],
                                     class_weight=model_values['class_weight_strategy']
                                    )
                                )
                            ]
                        )


clf.fit(x_train, y_train)

val_proba = clf.predict_proba(x_val)[:, 1]

best_threshold = threshold_parametrs['best_threshold']
best_bal_acc = 0.0

for thr in np.arange(values_threshold['start_threshold'],
                     values_threshold['stop_threshold'],
                     values_threshold['step_threshold']
                    ):
    val_pred = (val_proba >= thr).astype(int)
    bal_acc = balanced_accuracy_score(y_val, val_pred)
    
    if bal_acc > best_bal_acc:
        best_bal_acc = float(bal_acc)
        best_threshold = float(thr)

show_info(best_threshold, best_bal_acc)


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
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        f1=float(f1),
        auc_roc=float(auc),
        positive_rate=float(positive_rate),
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

write_metrics_reports(
    report_path=report_file,
    used_parametrs=model_values,
    model_name='Baseline Logistic Regression',
    train_rows=len(x_train),
    val_rows=len(x_val),
    test_rows=len(x_test),
    val_metrics=val_metrics_map,
    test_metrics=test_metrics_map,
    report_tickers=report_tickers
    )

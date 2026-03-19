# 04_baseline_hgb.py
# 


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
from additional_functions import (
    print_eval_block, write_metrics_reports,
    show_info, load_config, require_config_value
    )


project_root = Path(__file__).resolve().parents[1]
config = load_config(project_root)


data_file = project_root / Path(require_config_value(config, 'paths.dataset_parquet'))
model_values = require_config_value(config, 'models.high_gradient_boosting')

split_values = require_config_value(config, 'split')
training_ratio = float(split_values['training_ratio'])
validation_ratio = float(split_values['validation_ratio'])
validation_end_ratio = training_ratio + validation_ratio

features = require_config_value(config, 'features')
features_columns = features['high_gradient_boosting']

threshold_parametrs = require_config_value(config, 'threshold_search')
values_threshold = threshold_parametrs['high_gradient_boosting']


df = pd.read_parquet(data_file)
df['date'] = pd.to_datetime(df['date'])


for column in features_columns + ['target']:
    if column not in df.columns:
        raise ValueError(f'Колонка {column} отсутствует')
    
numeric_columns = [col for col in features_columns if col != 'ticker']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
df[numeric_columns] = df[numeric_columns].clip(-1e6, 1e6)
df = df.dropna(subset=numeric_columns + ['target']).copy()


df = df.sort_values('date').reset_index(drop=True)
unique_dates = df['date'].unique()
n_dates = len(unique_dates)


cut1 = int(n_dates * training_ratio)
cut2 = int(n_dates * validation_end_ratio)


train_end = unique_dates[cut1 - 1]
val_end = unique_dates[cut2 - 1]


train_df = df[df['date'] <= train_end].copy()
val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
test_df = df[df['date'] > val_end].copy()

x_train, y_train = train_df[features_columns], train_df['target']
x_val, y_val = val_df[features_columns], val_df['target']
x_test, y_test = test_df[features_columns], test_df['target']


numeric_features = numeric_columns.copy()
categorical_features = features['categorical']

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
    learning_rate=model_values['learning_rate'],
    max_depth=model_values['maximum_tree_depth'],
    max_iter=model_values['maximum_iterations'],
    min_samples_leaf=model_values['minimum_samples_per_leaf'],
    random_state=model_values['random_state_seed']
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
best_threshold = threshold_parametrs['best_threshold']
best_bal_acc = 0.0


for thr in np.arange(values_threshold['start_threshold'],
                     values_threshold['stop_threshold'],
                     values_threshold['step_threshold']):
    pred = (val_proba >= thr).astype(int)
    bal_acc = balanced_accuracy_score(y_val, pred)
    if bal_acc > best_bal_acc:
        best_bal_acc = float(bal_acc)
        best_threshold = float(thr)

show_info(best_threshold, best_bal_acc)


def evaluate(name: str,
             x_part: pd.DataFrame,
             y_part: pd.Series,
             threshold: float = threshold_parametrs['best_threshold']):
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
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        f1=float(f1),
        auc_roc=float(auc),
        positive_rate=float(positive_rate),
        y_part=y_part,
        pred=pred
        )
    
    return acc, bal_acc, f1, auc
    

val_metrics = evaluate('Validation', x_val, y_val, best_threshold)
test_metrics = evaluate('Test', x_test, y_test, best_threshold)

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


report_path = project_root / Path(require_config_value(config, 'paths.hgb_report'))
write_metrics_reports(
    report_path=report_path,
    used_parametrs=model_values,
    model_name='Baseline HighGradientBoosting',
    train_rows=len(x_train),
    val_rows=len(x_val),
    test_rows=len(x_test),
    val_metrics=val_metrics_map,
    test_metrics=test_metrics_map
    )
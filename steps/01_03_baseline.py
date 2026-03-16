# Первая разработка и тестирование
# Среднее значение показло 0.50 +- точности
# Что крайне не подходит под задачи проекта
# По своей сути данный скрипт и модель - это просто монетка, которая бросается с шансом 50 на 50

import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,\
                            classification_report, confusion_matrix


root = Path(__file__).resolve().parents[1]
data_path = root / 'data' / 'dataset.parquet'
reports_dir = root / 'reports'
reports_dir.mkdir(exist_ok=True)


df = pd.read_parquet(data_path)
df['date'] = pd.to_datetime(df['date'])

if 'target' not in df.columns:
    raise ValueError('Отсутствует колонка target в датасете')


feature_columns =   [
                    'ticker', 'open', 'high', 'low', 'close', 'volume',
                    'returns_1', 'returns_3', 'returns_5', 'returns_10',
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
    

numeric_features =   [
                    'open', 'high', 'low', 'close', 'volume',
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


def evaluate(name: str, x_part: pd.DataFrame, y_part: pd.Series):
    
    pred = clf.predict(x_part)
    proba = clf.predict_proba(x_part)[:, 1]
    
    
    acc = accuracy_score(y_part, pred)
    f1  = f1_score(y_part, pred)
    auc = roc_auc_score(y_part, proba)
    
    
    print(f"""
            ==== {name} ====
          
Accuracy: {acc:.4f}
F1: {f1:.4f}
ROC-AUC: {auc:.4f}


            Confusion matrix
        
{confusion_matrix(y_part, pred)}


            Classification report
        
{classification_report(y_part, pred, digits=4)}          
          """)
    
    return acc, f1, auc
    

val_metrics = evaluate('Validation', x_val, y_val)
test_metrics = evaluate('Test', x_test, y_test)


report_path = reports_dir / 'baseline_metrics.txt'
with open(report_path, 'w', encoding='utf-8') as file_out:
    
    file_out.write(f""" 
Baseline Logistic Regression
Train rows: {len(x_train)}
Val rows: {len(x_val)}
Test rows: {len(x_test)}


Validation - Accuracy: {val_metrics[0]:.4f}
F1: {val_metrics[1]:.4f}
AUC-ROC: {val_metrics[2]:.4f}


Test - Accuracy: {test_metrics[0]:.4f}
F1: {test_metrics[1]:.4f}
AUC-ROC: {test_metrics[2]:.4f}
""")
    
print(f'Report saved: {report_path}')
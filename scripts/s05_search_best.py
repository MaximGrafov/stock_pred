# 06_search_best_grid.py
#


import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

from sklearn.model_selection import ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score
    )

from support_functions import(
    load_config, print_eval_block, write_metrics_reports,
    show_info, require_config_value
    )



def prepare_xy(features_columns):
    
    for column in features_columns + ['target']:
        if column not in df.columns:
            raise ValueError(f'Колонка {column} отсутствует')
        
        
    x_train = train_df[features_columns].copy()
    y_train = train_df['target'].copy()
    
    x_val = val_df[features_columns].copy()
    y_val = val_df['target'].copy()
    
    x_test = test_df[features_columns].copy()
    y_test = test_df['target'].copy()
    
    
    numeric_columns = [column for column in features_columns if column != 'ticker']
    
    
    for part in [x_train, x_val, x_test]:
        part[numeric_columns] = part[numeric_columns].apply(pd.to_numeric, errors='coerce')
        part[numeric_columns] = part[numeric_columns].replace([-np.inf, np.inf], np.nan)
        part[numeric_columns] = part[numeric_columns].clip(-1e6, 1e6)
        
    
    return x_train, y_train, x_val, y_val, x_test, y_test, numeric_columns


def build_prepocessor(numeric_columns, use_scaler):
    
    num_steps: list[tuple[str, Any]] = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaler:
        num_steps.append(('scaler', StandardScaler()))
        
    
    numeric_transformer = Pipeline(steps=num_steps)
    categorical_transofrmer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )
    
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transofrmer, features['categorical'])
        ], sparse_threshold=0.0
    )
    
    
def find_best_threshold(y_val, val_proba, threshold_cfg):
    
    best_thr = threshold_params['best_threshold']
    best_bal = 0.0
    
    for thr in np.arange(
        threshold_cfg['start_threshold'],
        threshold_cfg['stop_threshold'],
        threshold_cfg['step_threshold']
    ):
        
        pred = (val_proba > thr).astype(int)
        bal = balanced_accuracy_score(y_val, pred)
        
        if bal > best_bal:
            best_bal = float(bal)
            best_thr = float(thr)
            
    return best_thr, best_bal


def evaluate(clf, name, x_part, y_part, threshold):
    
    proba = clf.predict_proba(x_part)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    acc = accuracy_score(y_part, pred)
    bal_acc = balanced_accuracy_score(y_part, pred)
    f1 = f1_score(y_part, pred)
    auc_roc = roc_auc_score(y_part, proba)
    positive_rate = pred.mean()
    
    
    print_eval_block(
        name=name,
        threshold=threshold,
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        f1=float(f1),
        auc_roc=float(auc_roc),
        positive_rate=positive_rate,
        y_part=y_part,
        pred=pred
        )
    
    
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'f1': float(f1),
        'auc_roc': float(auc_roc)
        }
    
    
def run_grid(
    model_name,
    feature_key,
    threshold_key,
    grid_params,
    model_factory,
    use_scaler,
    report_path_key
    ):
    
    feature_columns = features[feature_key]
    
    x_train, y_train, x_val, y_val, x_test, y_test, numeric_columns = prepare_xy(feature_columns)
    preprocessor = build_prepocessor(numeric_columns, use_scaler)

    to_numpy = FunctionTransformer(
             lambda x: x.to_numpy() if hasattr(x, 'to_numpy') else x,
             accept_sparse=True
            )
    
    threshold_cfg = threshold_params[threshold_key]
    
    
    best = None
    best_score = -1.0
    best_tie_bal = -1.0
    
    
    all_params = list(ParameterGrid(grid_params))
    print(f'\n[{model_name}] Вариантов: {len(all_params)}')
    
    for num, params in enumerate(all_params, start=1):

        model = model_factory(params)
        clf = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('to_numpy', to_numpy),
                ('model', model)
                ]
            )
        
        clf.fit(x_train, y_train)
        val_proba = clf.predict_proba(x_val)[:, 1]
        threshold, val_bal = find_best_threshold(y_val, val_proba, threshold_cfg)
        val_auc = roc_auc_score(y_val, val_proba)
        
        
        score = float(val_auc)
        
        
        if (score > best_score) or (score == best_score and val_bal > best_tie_bal):
            best_score = score
            best_tie_bal = val_bal
            best = {
                'params': params,
                'threshold': threshold,
                'val_auc': float(val_auc),
                'val_bal': float(val_bal),
                'clf': clf
                }
            
            
            
        print(f'[{model_name}] {num}/{len(all_params)}  |  val_auc={val_auc:.4f}  |  val_bal={val_bal:.4f}')
        
    if best is None:
                raise RuntimeError('Ошибка с best!')
              
    show_info(best['threshold'], best['val_bal'])    
    print(f'[{model_name}] Лучшие параметры: {best["params"]}')
    
    
    val_metrics = evaluate(best['clf'], 'Validation', x_val, y_val, best['threshold'])
    test_metrics = evaluate(best['clf'], 'Test', x_test, y_test, best['threshold'])
    
    report_path = project_root / Path(require_config_value(config, report_path_key))
    
    write_metrics_reports(
        report_path=report_path,
        model_name=f'{model_name} Grid Search',
        used_parametrs=best['params'],
        train_rows=len(x_train),
        val_rows=len(x_val),
        test_rows=len(x_test),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        report_tickers=report_tickers
        )
    


if __name__ == '__main__':
    
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)


    data_file = project_root / Path(require_config_value(config, 'paths.dataset_parquet'))
    df = pd.read_parquet(data_file)
    df['date'] = pd.to_datetime(df['date'])


    split_values = require_config_value(config, 'split')
    train_ratio = float(split_values['training_ratio'])
    validation_ratio = float(split_values['validation_ratio'])
    validation_end_ratio = train_ratio + validation_ratio

    threshold_params = require_config_value(config, 'threshold_search')
    features = require_config_value(config, 'features')
    models_grid = require_config_value(config, 'models_grid')


    df = df.sort_values('date').reset_index(drop=True)
    unique_date = df['date'].unique()
    n_dates = len(unique_date)


    cut1 = int(n_dates * train_ratio)
    cut2 = int(n_dates * validation_end_ratio)


    train_end = unique_date[cut1 - 1]
    val_end = unique_date[cut2 - 1]

    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] > val_end].copy()

    report_tickers = sorted(train_df['ticker'].dropna().astype(str).unique().tolist())

    
    run_grid(
        model_name='Logistic Regression',
        feature_key='baseline',
        threshold_key='logistic_regression_baseline',
        grid_params=models_grid['logistic_regression'],
        model_factory=lambda p: LogisticRegression(
            max_iter=p['maximum_iterations'],
            n_jobs=p['number_of_jobs'],
            class_weight=p['class_weight_strategy']
            ),
        use_scaler=True,
        report_path_key='paths.baseline_report'
        )


    run_grid(
        model_name='High Gradient Boosting',
        feature_key='hgb_lgb_cat',
        threshold_key='high_gradient_boosting',
        grid_params=models_grid['high_gradient_boosting'],
        model_factory=lambda p: HistGradientBoostingClassifier(
            learning_rate=p['learning_rate'],
            max_depth=p['maximum_tree_depth'],
            max_iter=p['maximum_iterations'],
            min_samples_leaf=p['minimum_samples_per_leaf'],
            random_state=p['random_state_seed']
            ),
        use_scaler=False,
        report_path_key='paths.hgb_report'
        )


    run_grid(
        model_name='Light Gradient Boosting',
        feature_key='hgb_lgb_cat',
        threshold_key='light_gradient_boosting',
        grid_params=models_grid['light_gradient_boosting'],
        model_factory=lambda p: LGBMClassifier(
            learning_rate=p['learning_rate'],
            n_estimators=p['n_estimators'],
            num_leaves=p['num_leaves'],
            min_child_samples=p['min_child_samples'],
            random_state=p['random_state'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'],
            max_depth=p['max_depth'],
            verbosity=p['verbosity']
            ),
        use_scaler=False,
        report_path_key='paths.lightgbm_report'
        )
    
    
    run_grid(
        model_name='Cat Boost Classifier',
        feature_key='hgb_lgb_cat',
        threshold_key='cat_boost_classifier',
        grid_params=models_grid['cat_boost_classifier'],
        model_factory=lambda p: CatBoostClassifier(
            learning_rate=p['learning_rate'],
            depth=p['depth'],
            iterations=p['iterations'],
            l2_leaf_reg=p['l2_leaf_reg'],
            random_seed=p['random_seed'],
            loss_function=p['loss_function'],
            eval_metric=p['eval_metric'],
            verbose=p['verbose']
            ),
        use_scaler=False,
        report_path_key='paths.cat_report'
        )
# additional_functions.py
# Файл для скриптов, которые создают лишний визуальный мусор в файлах


import yaml
import pandas as pd
import numpy as np

from datetime import datetime
from typing import Any, Mapping
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


def print_eval_block(
    name: str,
    threshold: float,
    accuracy: float,
    balanced_accuracy: float,
    f1: float,
    auc_roc: float,
    positive_rate: float,
    y_part: pd.Series | np.ndarray,
    pred: pd.Series | np.ndarray
    ) -> None:
    
    print(f"""
            ==== {name} ====

Threshold: {threshold:.4f}
Accuracy: {accuracy:.4f}
Balanced accuracy: {balanced_accuracy:.4f}
F1: {f1:.4f}
ROC-AUC: {auc_roc:.4f}
Predicted class 1 share: {positive_rate:.4f}


            Confusion matrix
        
{confusion_matrix(y_part, pred)}


            Classification report
        
{classification_report(y_part, pred, digits=4)}          
          """)
    
    
def write_metrics_reports(
    report_path: str | Path,
    trial_report_path: str | Path,
    used_parametrs: dict,
    model_name: str,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    val_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float]    
    ) -> None:
    
    path_report = Path(report_path)
    path_report.parent.mkdir(parents=True, exist_ok=True)
    
    path_trial_report = Path(trial_report_path)
    path_trial_report.parent.mkdir(parents=True, exist_ok=True)
    
    model_parametrs = chr(10).join(f'{key}: {value}' for key, value in used_parametrs.items())
        
    with path_report.open('w', encoding='utf-8') as last_report_file,\
         path_trial_report.open('w+', encoding='utf-8') as trial_report_file:

        writer_text = f"""
            
            ==== {model_name} ====
            
Train rows: {train_rows}
Val rows: {val_rows}
Test rows: {test_rows}


            ==== Used parametrs ====
            
{model_parametrs}


            ==== Validation metrics ====

Validation - Accuracy: {val_metrics['accuracy']:.4f}
Balanced accuracy: {val_metrics['balanced_accuracy']:.4f}
F1: {val_metrics['f1']:.4f}
AUC-ROC: {val_metrics['auc_roc']:.4f}


            ==== Test metrics ====

Test - Accuracy: {test_metrics['accuracy']:.4f}
Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}
F1: {test_metrics['f1']:.4f}
AUC-ROC: {test_metrics['auc_roc']:.4f}

"""
        
        last_report_file.write(writer_text)
        trial_report_file.write(writer_text)
    
        print(f'Reports saved: \n{report_path} \n{trial_report_path}')
        
        
       
        
def load_config(project_root: Path) -> dict:
    config_path = project_root / 'config.yaml'
    
    with config_path.open('r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file) or {}
    
    return config


def require_config_value(config: Mapping[str, Any], key_path: str) -> Any:
    current: Any = config
    
    for part in key_path.split('.'):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f'В файле config отсутствует ключ: {key_path}')
       
        current = current[part]
    
    if current is None:
        raise KeyError(f'В файле config пустой ключ: {key_path}')
    
    return current


def show_info(threshold, bal_acc):
    print(f'Best threshold on VAL: {threshold:.2f}, accuarcy: {bal_acc:.4f}')
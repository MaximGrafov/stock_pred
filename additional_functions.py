# additional_functions.py
# Файл для скриптов, которые создают лишний визуальный мусор в файлах

import pandas as pd
import numpy as np

from typing import Mapping
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
          """.strip())
    
    
def write_metrics_reports(
    report_path: str | Path,
    model_name: str,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    val_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float]    
    ) -> None:
    
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as file_out:
    
        file_out.write(f"""
            
            ==== {model_name} ====
            
Train rows: {train_rows}
Val rows: {val_rows}
Test rows: {test_rows}


            ==== Validation metrics ====

Validation - Accuracy: {val_metrics['accuracy']:.4f}
Balanced accuracy: {val_metrics['balanced_accuracy']:4f}
F1: {val_metrics['f1']:.4f}
AUC-ROC: {val_metrics['auc_roc']:.4f}


            ==== Test metrics ====

Test - Accuracy: {test_metrics['accuracy']:.4f}
Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}
F1: {test_metrics['f1']:.4f}
AUC-ROC: {test_metrics['auc_roc']:.4f}

""")
    
        print(f'Report saved: {report_path}')
#
#

import copy
import subprocess
import sys
import yaml

from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'config.yaml'


DATASET_GRID = [

    {"target_horizon_days": 10, "target_absolute_minimum_return": 0.03},
    {"target_horizon_days": 10, "target_absolute_minimum_return": 0.05},
    {"target_horizon_days": 10, "target_absolute_minimum_return": 0.07},
    {"target_horizon_days": 10, "target_absolute_minimum_return": 0.10},
    

    {"target_horizon_days": 20, "target_absolute_minimum_return": 0.03},
    {"target_horizon_days": 20, "target_absolute_minimum_return": 0.05},
    {"target_horizon_days": 20, "target_absolute_minimum_return": 0.07},
    {"target_horizon_days": 20, "target_absolute_minimum_return": 0.10},

    {"target_horizon_days": 25, "target_absolute_minimum_return": 0.03},
    {"target_horizon_days": 25, "target_absolute_minimum_return": 0.05},
    {"target_horizon_days": 25, "target_absolute_minimum_return": 0.07},
    {"target_horizon_days": 25, "target_absolute_minimum_return": 0.10},

    {"target_horizon_days": 30, "target_absolute_minimum_return": 0.03},
    {"target_horizon_days": 30, "target_absolute_minimum_return": 0.05},
    {"target_horizon_days": 30, "target_absolute_minimum_return": 0.07},
    {"target_horizon_days": 30, "target_absolute_minimum_return": 0.10},
    ]


def run_script(rel_path: str | Path) -> None:
    
    subprocess.run([sys.executable, str(ROOT / rel_path)], check=True)
    
    
def main() -> None:
    
    original_text = CONFIG_PATH.read_text(encoding='utf-8')
    base_cfg = yaml.safe_load(original_text)
    
    
    try:
        for combo in DATASET_GRID:
            h = combo['target_horizon_days']
            r = combo['target_absolute_minimum_return']
            r_tag = str(r).replace('.', '')
            
            
            cfg = copy.deepcopy(base_cfg)
            
            
            cfg['dataset']['target_horizon_days'] = h
            cfg['dataset']['target_absolute_minimum_return'] = r
            
            
            cfg['paths']['dataset_parquet'] = f'data/dataset_h{h}_r{r_tag}.parquet'
            cfg['paths']['baseline_report'] = f'reports/grid_runs/baseline_h{h}_r{r_tag}.txt'
            cfg['paths']['hgb_report'] = f'reports/grid_runs/hgb_h{h}_r{r_tag}.txt'
            cfg['paths']['lightgbm_report'] = f'reports/grid_runs/lightgbm_h{h}_r{r_tag}.txt'
            
            
            CONFIG_PATH.write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
            
            
            print(f'\n ==== RUN h={h} r={r} ====')
            run_script('scripts/02_make_dataset.py')
            run_script('scripts/06_search_best_grid.py')
            
    finally:
        CONFIG_PATH.write_text(original_text, encoding='utf-8')
        print('config.yaml восстановлен')
        
        
if __name__ == '__main__':
    main()
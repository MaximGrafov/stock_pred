Последовательность команд:

git clone (ссылка)
	|
python -m venv .venv
	|
source .venv/scripts/activate
	|
pip install -r requirements.txt
	|
export PYTHONPATH=$(pwd)
	|
python create_db.py
	|
python fill_db.py
	|
python scripts/01_checking_db.py
	|
python scripts/02_make_dataset.py
	|
python scripts/03_baseline.py
	|
python scripts/04_baseline_hgb.py
	|
python scripts/05_lightgbm.py
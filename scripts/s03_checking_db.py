# 01_checking_db.py
# Скрипт проверяющий корректность заполнения БД

import sqlite3
import pandas as pd

from pathlib import Path

from additional_functions import load_config, require_config_value

project_root = Path(__file__).resolve().parents[1]
config = load_config(project_root)


database_file = require_config_value(config, 'paths.database_file')
database_path = project_root / database_file

with sqlite3.connect(database_path) as connect:
    total_rows = pd.read_sql_query("SELECT COUNT(*) AS count FROM stock_prices", connect)
    print(f'В файле {database_path.name}: {total_rows["count"][0]} строк')


    by_ticker = pd.read_sql_query(
                                """--sql
                                    SELECT 
                                        ticker, 
                                        COUNT(*)  AS rows_count,
                                        MIN(date) AS min_date,
                                        MAX(date) AS max_date
                                    FROM stock_prices
                                    GROUP BY ticker
                                    ORDER BY rows_count DESC                   
                                """, connect)


    print(f"""\n
    По тикерам:
    {by_ticker.to_string(index=False)}
    """)


    duplicates = pd.read_sql_query(
                                """--sql
                                    SELECT COUNT(*) AS dup_count
                                        FROM
                                            (
                                            SELECT date, ticker, COUNT(*) AS cou
                                            FROM stock_prices
                                            GROUP BY date, ticker
                                            HAVING cou > 1
                                            )

                                    """, connect)


    print(f"\nНайден дубликатов: {int(duplicates['dup_count'][0])}")


    nulls = pd.read_sql_query("""--sql
                                SELECT
                                    SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END)   AS null_open,
                                    SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END)   AS null_high,
                                    SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END)    AS null_low,
                                    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END)  AS null_close,
                                    SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume
                                FROM stock_prices
                                """, connect)


    print(f"""
    Пропуски:
    {nulls.to_string(index=False)}
          """)
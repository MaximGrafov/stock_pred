# complete_all.py
# Файл создан для сборки и порядкового запуска скриптов

import subprocess
import sys
import os

from pathlib import Path

root = Path(__file__).resolve().parent

script_files = sorted([entry.path for entry in os.scandir('scripts') if entry.is_file() and entry.name.endswith('.py')])


for script in script_files:
    print(f'\nЗапущен скрипт файла: {script}\n')

    subprocess.run([sys.executable, str(root / script)], check=True)
    
    print(f'\nСкрипт файла {script} завершил работу\n')
    
    
print('Все скрипты из файлов отработаны')
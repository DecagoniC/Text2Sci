"""
Скрипт для автоматического запуска всех тестов из каталога src/unit_tests.
Ищет тестовые файлы по шаблону test_*.py и запускает их через pytest.
"""

import subprocess
import sys
from pathlib import Path

tests_dir = Path("src/unit_tests")

if not tests_dir.exists():
    print("[!] Папка с тестами не найдена: src/unit_tests")
    sys.exit(1)

print("[+] Поиск тестовых файлов...\n")

test_files = sorted(tests_dir.rglob("test_*.py"))

if not test_files:
    print("[!] Не найдено файлов вида test_*.py")
    sys.exit(1)

print("[+] Найдены тесты:")
for f in test_files:
    print("   •", f)

print("\n[+] Запуск pytest...\n")

result = subprocess.run(
    ["pytest", "-q", str(tests_dir)],
    text=True
)

sys.exit(result.returncode)
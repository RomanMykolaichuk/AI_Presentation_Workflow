"""Координаційний скрипт для запуску повного конвеєра.

Приклад використання
--------------------
>>> import main
>>> main.main()  # далі слідуйте інструкціям у консолі
"""
from __future__ import annotations

from collect_data import collect
from fill_template import build_pptx
from generate_structure import generate


def main() -> None:
    topic = input("Введіть тему презентації: ")
    results_file = collect(topic)
    structure_file = generate(results_file)
    pptx_file = build_pptx(structure_file)
    print(f"Готово! Файл презентації: {pptx_file}")


if __name__ == "__main__":
    main()

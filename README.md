# thekursova: Ukrainian Poetry AI

Повноцінний пайплайн для збору корпусу українських віршів, донавчання мовної моделі та генерації римованих текстів без залежності від зовнішніх API. Пайплайн складається зі скрапінгу (опційно), збирання датасету, тренування та генерації.

## Модулі
- `poetry_ai.data` — чистка тексту, скрапінг сайтів (з перевіркою robots.txt, яку можна вимкнути при наявності дозволу), побудова Hugging Face `Dataset`.
- `poetry_ai.training` — налаштовує та запускає Hugging Face `Trainer` для авто-регресивної моделі.
- `poetry_ai.generation` — завантаження збереженої моделі та генерація віршів із перевіркою рими/складів.
- `poetry_ai.cli` — CLI з підкомандами `scrape`, `train`, `generate`.

## Вибір базової моделі
За замовчуванням використовується велика багатомовна модель `facebook/xglm-1.7B` (локально, без API) — вона краще відтворює українську морфологію та рими, ніж менші GPT-2 похідні. Якщо ресурсів мало, можна переключитись на компактніші варіанти (`facebook/xglm-564M`, `google/mt5-small`, `ai-forever/rugpt3medium_based_on_gpt2`) через параметр `--model-name`.

## Встановлення
Щоб команда `python -m poetry_ai.cli ...` працювала з будь-якої директорії без помилки
`ModuleNotFoundError: No module named 'poetry_ai'`, встановіть пакет в editable-режимі:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # реєструє poetry_ai як встановлений пакет
```

Альтернатива без встановлення — запускати команди з кореня репозиторію або додати
`PYTHONPATH=$(pwd)` перед викликом (`PYTHONPATH=$(pwd) python -m poetry_ai.cli ...`).

## Скрапінг (опційно)
```bash
# Найзручніше — використати готовий профіль парсера для poetryclub.com.ua
python -m poetry_ai.cli scrape --preset poetryclub --start-page 1 --end-page 5 --output scraped_poems.json

# Якщо robots.txt тимчасово недоступний або у вас є дозвіл — можна вимкнути перевірку
# python -m poetry_ai.cli scrape --preset poetryclub --ignore-robots --output scraped_poems.json

# За потреби профіль можна перевизначити адресою та селекторами
# python -m poetry_ai.cli scrape \
#   https://poetryclub.com.ua/listpoems.php \
#   "div.vers" \
#   "div.vers > p" \
#   --title-selector "div.vers > h3" \
#   --start-page 1 --end-page 5 \
#   --ignore-robots \
#   --output scraped_poems.json
```
Скрапер перевіряє `robots.txt`; якщо файл недоступний або повертає помилку, виконується обережний прохід із попередженням. Можна явно вимкнути перевірку через `--ignore-robots` (або `obey_robots=False` у коді) лише за наявності дозволу. Профіль `poetryclub` використовує реальний каталог `https://poetryclub.com.ua/listpoems.php` із блоками `div.vers` і заголовком `h3`. Якщо структура сторінки зміниться, можна передати власні селектори або створити новий профіль у `SCRAPER_PRESETS`.

> Порада: запускайте CLI через `python -m poetry_ai.cli ...`, щоб уникнути помилки `ImportError: attempted relative import with no known parent package` при прямому виклику файлів.

## Тренування
```bash
python -m poetry_ai.cli train \
  --dataset staliuk/ukrainian-poetry \
  --scraped scraped_poems.json \
  --model-name facebook/xglm-1.7B \
  --output-dir poetry-model \
  --max-length 256 --train-batch-size 2 --num-train-epochs 3
```

## Генерація
```bash
python -m poetry_ai.cli generate "Осінній вечір над містом" \
  --model-path poetry-model \
  --lines 4 --rhyme-scheme ABAB --expected-syllables 10
```
CLI виводить вірш у рамці, підписує римну групу, суфікс рими та підрахунок складів у кожному рядку.

## Перевірки якості
- Скрапінг: 3–5 тестових сторінок, обробка помилок HTTP.
- Тренування: прогін на кількох десятках зразків для перевірки токенізації.
- Генерація: перевірка рими та кількості складів на отриманих рядках.

## Примітки
- Пайплайн працює повністю локально; інтернет потрібен лише для завантаження моделі/датасету або скрапінгу.
- Для швидкого старту використовуйте готовий датасет `staliuk/ukrainian-poetry` без скрапінгу.

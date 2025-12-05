# thekursova: Ukrainian Poetry AI

Повноцінний пайплайн для збору корпусу українських віршів, донавчання мовної моделі та генерації римованих текстів без залежності від зовнішніх API. Пайплайн складається зі скрапінгу (опційно), збирання датасету, тренування та генерації.

## Модулі
- `poetry_ai.data` — чистка тексту, скрапінг сайтів (з перевіркою robots.txt), побудова Hugging Face `Dataset`.
- `poetry_ai.training` — налаштовує та запускає Hugging Face `Trainer` для авто-регресивної моделі.
- `poetry_ai.generation` — завантаження збереженої моделі та генерація віршів із перевіркою рими/складів.
- `poetry_ai.cli` — CLI з підкомандами `scrape`, `train`, `generate`.

## Вибір базової моделі
За замовчуванням використовується `ai-forever/rugpt3small_based_on_gpt2` (локально, без API). За потреби замініть на україномовну/багатомовну модель (наприклад, `facebook/xglm-564M` або `google/mt5-small`) у параметрі `--model-name`.

## Встановлення
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Скрапінг (опційно)
```bash
python -m poetry_ai.cli scrape \
  https://poetryclub.com.ua/metrs/index.php \
  "div.vers" \
  "div.vers > p" \
  --title-selector "div.vers > h3" \
  --start-page 1 --end-page 3 \
  --output scraped_poems.json
```
Скрапер перевіряє `robots.txt`; якщо сайт забороняє обходи, буде кинуто помилку. Приклад вище використовує реальний сайт з каталогом віршів `poetryclub.com.ua` (розмітка актуальна станом на 2024‑12, блок вірша має `div.vers` із параграфами всередині). Якщо структура сторінки зміниться, підберіть актуальні CSS‑селектори.

## Тренування
```bash
python -m poetry_ai.cli train \
  --dataset staliuk/ukrainian-poetry \
  --scraped scraped_poems.json \
  --model-name ai-forever/rugpt3small_based_on_gpt2 \
  --output-dir poetry-model \
  --max-length 256 --train-batch-size 2 --num-train-epochs 3
```

## Генерація
```bash
python -m poetry_ai.cli generate "Осінній вечір над містом" \
  --model-path poetry-model \
  --lines 4 --rhyme-scheme ABAB --expected-syllables 10
```

## Перевірки якості
- Скрапінг: 3–5 тестових сторінок, обробка помилок HTTP.
- Тренування: прогін на кількох десятках зразків для перевірки токенізації.
- Генерація: перевірка рими та кількості складів на отриманих рядках.

## Примітки
- Пайплайн працює повністю локально; інтернет потрібен лише для завантаження моделі/датасету або скрапінгу.
- Для швидкого старту використовуйте готовий датасет `staliuk/ukrainian-poetry` без скрапінгу.

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
# Використайте готовий профіль для onlyart.org.ua.
# Пагінація WordPress має вигляд /page/{n}/, тому базова сторінка береться без query,
# а наступні сторінки формуються автоматично за шаблоном (page_template).
# Скрапер переходить за посиланнями статей (h2.entry-title a) і дістає повний текст із сторінки вірша.
python -m poetry_ai.cli scrape --preset onlyart --start-page 1 --end-page 5 --output scraped_poems.json

# Якщо robots.txt тимчасово недоступний або у вас є дозвіл — можна вимкнути перевірку
# python -m poetry_ai.cli scrape --preset onlyart --ignore-robots --output scraped_poems.json

# За потреби профіль можна перевизначити адресою та селекторами
# python -m poetry_ai.cli scrape \
#   https://onlyart.org.ua/category/ukrayinski-poety \
#   "article" \
#   ".entry-content p" \
#   --page-template "{base}/page/{page}/" \
#   --title-selector ".entry-title" \
#   --start-page 1 --end-page 5 \
#   --ignore-robots \
#   --output scraped_poems.json
```
Скрапер перевіряє `robots.txt`; якщо файл недоступний або повертає помилку, виконується обережний прохід із попередженням. Можна явно вимкнути перевірку через `--ignore-robots` (або `obey_robots=False` у коді) лише за наявності дозволу. Профіль `onlyart` оперує категорією `https://onlyart.org.ua/category/ukrayinski-poety` з WordPress-розміткою: спочатку знаходить `article`, бере посилання з `h2.entry-title a`/`header.entry-header a`, відкриває повну сторінку вірша і читає параграфи `.entry-content p` (із заголовком `.entry-title`). Якщо структура сторінки зміниться, передайте власні селектори або створіть новий профіль у `SCRAPER_PRESETS`, зокрема через новий параметр `--page-template` для коректної побудови посилань пагінації. Для poesia.org.ua та poetryclub.com.ua збережено альтернативні профілі, але вони не використовуються за замовчуванням.

> Порада: запускайте CLI через `python -m poetry_ai.cli ...`, щоб уникнути помилки `ImportError: attempted relative import with no known parent package` при прямому виклику файлів.

## Тренування
```bash
# Навчання лише на локальних/скраплених віршах (значення за замовчуванням)
python -m poetry_ai.cli train \
  --dataset none \
  --scraped manual_poems.json \
  --model-name facebook/xglm-1.7B \
  --output-dir poetry-model-local

# Або використання публічного датасету з Hugging Face (наприклад, syvin/ukrainian-literature)
python -m poetry_ai.cli train \
  --dataset syvin/ukrainian-literature \
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

## Ручне додавання віршів (якщо сайт змінює розмітку)
Якщо жоден профіль скрапінгу не спрацьовує або сайт змінив структуру, запишіть тексти вручну у файл `manual_poems.json` у корені репозиторію. Формат — масив об'єктів з полями `text`, `title`, `author`, `url` (URL можна лишити порожнім). У файлі вже додано кілька класичних українських поезій (Шевченко, Леся Українка, Франко) як приклад.

Після заповнення передайте шлях до файлу у `--scraped` під час тренування або об'єднайте з даними, зібраними скрапером (обидва файли мають ідентичний формат). Прапор `--dataset none` дозволяє навчатися виключно на цих локальних записах без Hugging Face датасетів (це значення використовується за замовчуванням, щоб уникати помилок доступу до недоступних датасетів).

## Перевірки якості
- Скрапінг: 3–5 тестових сторінок, обробка помилок HTTP.
- Тренування: прогін на кількох десятках зразків для перевірки токенізації.
- Генерація: перевірка рими та кількості складів на отриманих рядках.

## Примітки
- Пайплайн працює повністю локально; інтернет потрібен лише для завантаження моделі/датасету або скрапінгу.
- Для швидкого старту можна використати доступний датасет, наприклад `syvin/ukrainian-literature`, або обмежитися локальними файлами (`--dataset none`).

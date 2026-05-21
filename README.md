# Pain Detection Monitor

Проєкт для оцінювання болю за зображенням, відео з камери та face/pose landmarks. У папці є два підходи:

- нейромережевий класифікатор `Pain` / `NoPain` на базі `ResNet18`;
- rule-based моніторинг через MediaPipe Face/Pose landmarks і мікрорухи.

> Проєкт не є медичним діагностичним інструментом. Результати варто розглядати як експериментальну оцінку для навчальних або дослідницьких задач.

## Структура проєкту

```text
.
├── train.py                 # навчання ResNet18 на SynPain
├── infer.py                 # інференс на зображенні або webcam
├── monitor.py               # timed-моніторинг нейромережевої моделі
├── pain_mediapipe.py        # rule-based оцінка болю за face landmarks
├── pain_multimodal.py       # face + body micro-motion моніторинг
├── requirements.txt         # залежності Python
├── models/
│   └── pain_resnet18.pt     # збережені ваги ResNet18
├── data/
│   ├── SynPain_Part*/       # розпакований датасет SynPain
│   ├── face_detector/       # OpenCV DNN face detector
│   └── mediapipe/           # MediaPipe task-файли
└── src/
    ├── data.py              # завантаження/підготовка SynPain
    ├── face_detector.py     # детекція та crop обличчя
    ├── mediapipe_face.py    # MediaPipe Face Landmarker wrapper
    ├── mediapipe_pose.py    # MediaPipe Pose Landmarker wrapper
    ├── micro_motion.py      # розрахунок мікрорухів
    ├── pain_rules.py        # rule-based ознаки болю
    ├── overlay.py           # відмальовування HUD/skeleton
    ├── visibility_filter.py # фільтр видимості обличчя
    └── hair_filter.py       # фільтр перекриття волоссям
```

## Встановлення

Рекомендовано використовувати віртуальне середовище:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Якщо планується автоматичне завантаження датасету або face detector файлів через код, також потрібен пакет `requests`:

```bash
pip install requests
```

## Дані та моделі

Для навчання використовується SynPain. Скрипт `src/data.py` вміє завантажити та розпакувати дві частини датасету в `data/`, якщо папки `data/SynPain_Part1` і `data/SynPain_Part2` відсутні.

Для нейромережевого інференсу за замовчуванням використовуються ваги:

```text
models/pain_resnet18.pt
```

Для MediaPipe-моніторів потрібні task-файли:

```text
data/mediapipe/face_landmarker.task
data/mediapipe/pose_landmarker.task
```

## Навчання моделі

```bash
python train.py --data-dir data --epochs 3 --batch-size 32 --model-out models/pain_resnet18.pt
```

Корисні параметри:

- `--use-face` - навчати на crop обличчя через OpenCV DNN detector;
- `--face-conf 0.5` - confidence threshold для face detector;
- `--num-workers 2` - кількість workers для `DataLoader`;
- `--lr 1e-4` - learning rate.

## Інференс

На одному зображенні:

```bash
python infer.py --image path/to/image.jpg --weights models/pain_resnet18.pt
```

З webcam:

```bash
python infer.py --camera --camera-id 0 --weights models/pain_resnet18.pt
```

Вимкнення фільтрів:

```bash
python infer.py --camera --no-face --no-vis-filter --no-hair-filter
```

## Нейромережевий моніторинг

Скрипт `monitor.py` збирає ймовірності болю в часі та зберігає графік.

```bash
python monitor.py --duration 60 --interval 1.0 --output pain_monitor.png
```

Для запуску без live-вікна:

```bash
python monitor.py --duration 60 --no-display
```

Для тесту на папці зображень замість камери:

```bash
python monitor.py --image-dir path/to/images --duration 30 --no-display
```

## MediaPipe Face Monitor

Rule-based оцінка за face landmarks: брови, очі, ніс, рот і jaw opening.

```bash
python pain_mediapipe.py --duration 60 --output pain_mediapipe.png --csv-out pain_mediapipe.csv
```

Корисні параметри:

- `--calibrate 3.0` - перші секунди для нейтрального baseline;
- `--threshold 60` - поріг для графіка та summary;
- `--show-deltas` - показати відхилення facial features;
- `--draw-face-lines` - намалювати face skeleton overlay.

## Multimodal Monitor

`pain_multimodal.py` об'єднує rule-based face score і body/face micro-motion.

```bash
python pain_multimodal.py \
  --duration 60 \
  --output pain_multimodal.png \
  --csv-out pain_multimodal.csv \
  --npz-out pain_multimodal.npz
```

Корисні параметри:

- `--w-face 0.7` і `--w-body 0.3` - ваги face/body score;
- `--pose-min-quality 0.55` - мінімальна якість pose detection;
- `--draw-pose` - показати pose skeleton;
- `--draw-face-lines` - показати face skeleton;
- `--show-deltas` - показати facial feature deltas.

## Які файли виглядають зайвими

Ці файли не потрібні для вихідного коду і зазвичай не зберігаються в чистому репозиторії:

- `.DS_Store`, `data/.DS_Store` - системні файли macOS;
- `__pycache__/`, `src/__pycache__/` і `*.pyc` - Python bytecode cache;
- `pain_monitor.png`, `pain_mediapipe.png`, `pain_meadiapipe.png`, `pain_multimodal.png` - згенеровані графіки/скриншоти результатів;
- `pain_multimodal.csv`, `pain_multimodal.npz` - згенеровані результати моніторингу;
- `data/SynPain_Part1.zip`, `data/SynPain_Part2.zip` - архіви можна видалити, якщо датасет уже розпакований і повторне розпакування не потрібне.

Великі дані та ваги моделі краще не комітити у звичайний репозиторій без Git LFS або окремого сховища:

- `data/SynPain_Part1/`, `data/SynPain_Part2/`;
- `models/pain_resnet18.pt`.

## Рекомендований `.gitignore`

```gitignore
.DS_Store
__pycache__/
*.pyc

data/SynPain_Part*/
data/*.zip

*.png
*.csv
*.npz

models/*.pt
```

Якщо потрібно зберігати приклади графіків або готову модель прямо в проєкті, приберіть відповідні рядки з `.gitignore`.

# American Sign Language (ASL) Alphabet Recognition

Система распознавания букв американского языка жестов с использованием глубокого обучения. Проект реализует полный пайплайн от загрузки данных до продакшн-готового инференса с использованием современных MLOps практик.

## Описание проекта

Проект решает задачу классификации изображений рук, показывающих буквы американского языка жестов (ASL). Система способна распознавать 29 классов: 26 букв алфавита (A-Z) и три специальных символа (del, nothing, space).

### Основные возможности

- **Загрузка и управление данными** через DVC
- **Обучение моделей** с использованием PyTorch Lightning
- **Экспериментальное отслеживание** через MLflow
- **Экспорт моделей** в ONNX формат
- **Инференс** через ONNX Runtime или Triton Inference Server
- **Конфигурация** через Hydra с иерархическими конфигами
- **Автоматизация качества кода** через pre-commit hooks

### Архитектура

Проект поддерживает несколько архитектур моделей:

- **BaselineCNN**: Простая сверточная нейронная сеть
- **TransferResNet**: Transfer learning на основе ResNet (ResNet18, ResNet34, etc.)

## Setup

### Требования

- Python >= 3.10
- Poetry (для управления зависимостями)
- Git
- DVC (для управления данными)
- Docker и Docker Compose (для Triton и MLflow)

### Установка окружения

1. **Создайте виртуальное окружение и установите зависимости:**

   ```bash
   # Установите Poetry, если еще не установлен
   curl -sSL https://install.python-poetry.org | python3 -

   # Установите зависимости проекта
   poetry install

   # Установите плагин shell
   poetry self add poetry-plugin-shell

   # Активируйте окружение
   poetry shell
   ```

2. **Настройте pre-commit hooks:**

   ```bash
   pre-commit install
   ```

3. **Проверьте установку:**
   ```bash
   pre-commit run -a
   ```

### Структура проекта

```
.
├── asl_project/          # Основной Python пакет
│   ├── __init__.py
│   ├── commands.py       # CLI команды через Fire
│   ├── train.py          # Обучение модели
│   ├── data.py           # DataModule и Dataset
│   ├── model.py          # Архитектуры моделей
│   ├── infer.py          # Инференс
│   ├── export.py         # Экспорт в ONNX
│   ├── callbacks.py      # Callbacks для логирования
│   └── utils.py          # Утилиты
├── configs/              # Hydra конфигурации
│   ├── default.yaml
│   ├── train.yaml
│   ├── inference.yaml
│   ├── data/
│   │   └── default.yaml
│   └── model/
│       ├── baseline.yaml
│       └── resnet.yaml
├── scripts/              # Вспомогательные скрипты
│   ├── prepare_triton_model.py
│   └── test_triton_client.py
├── triton_models/        # Triton Inference Server модели
│   └── asl_model/
│       └── config.pbtxt
├── plots/                # Сохраненные графики метрик
├── models/               # Сохраненные модели и чекпоинты
├── data/                 # Данные (управляются через DVC)
├── pyproject.toml        # Зависимости Poetry
├── docker-compose.yml    # Docker Compose для MLflow и Triton
├── .pre-commit-config.yaml
└── README.md
```

## Train

### Загрузка данных

Данные автоматически загружаются через DVC при первом запуске обучения. Если данные отсутствуют локально, они будут скачаны из удаленного хранилища (S3) или напрямую из открытых источников (если настроено).

### Запуск обучения

Обучение запускается через единую точку входа `commands.py`:

```bash
# Базовое обучение с дефолтными параметрами
python -m asl_project.commands train

# Обучение с переопределением параметров
python -m asl_project.commands train model=baseline trainer.max_epochs=50

# Обучение ResNet модели
python -m asl_project.commands train model=resnet data.batch_size=32
```

### Конфигурация обучения

Основные параметры обучения настраиваются через конфиги в `configs/`:

- **Модель**: Выбирается через `model=baseline` или `model=resnet`
- **Данные**: Настраиваются в `configs/data/default.yaml`
- **Trainer**: Настраивается в `configs/train.yaml`

### Мониторинг обучения

Во время обучения:

- **MLflow**: Метрики, гиперпараметры и артефакты логируются в MLflow (по умолчанию `http://127.0.0.1:8080`)
- **Графики**: Сохраняются в `plots/` и логируются в MLflow:
  - `loss_curves.png` - кривые потерь
  - `accuracy_curves.png` - кривые точности
  - `f1_score.png` - F1 score
  - `metrics_overview.png` - обзор всех метрик

### Запуск MLflow UI

```bash
docker-compose up -d mlflow
# Откройте http://localhost:8080
```

## Production preparation

### Экспорт в ONNX

После обучения модель экспортируется в ONNX формат для оптимизированного инференса:

```bash
python -m asl_project.commands export checkpoint_path=models/checkpoints/best.ckpt
```

ONNX модель будет сохранена в `models/model.onnx` (или путь из конфига).

### Подготовка модели для Triton Inference Server

1. **Экспортируйте модель в ONNX** (см. выше)

2. **Подготовьте структуру Triton:**

   ```bash
   python scripts/prepare_triton_model.py
   ```

3. **Проверьте структуру:**
   ```
   triton_models/
   └── asl_model/
       ├── config.pbtxt
       └── 1/
           └── model.onnx
   ```

### Артефакты для продакшна

Для запуска инференса в продакшне необходимы:

- **ONNX модель** (`models/model.onnx`)
- **Конфигурация** (`configs/inference.yaml`)
- **Классы** (список из 29 классов ASL алфавита)
- **Препроцессинг** (нормализация с ImageNet статистиками)

Все эти компоненты включены в проект и настраиваются через конфиги.

## Infer

### Локальный инференс

Инференс запускается через единую точку входа:

```bash
# Инференс на одном изображении
python -m asl_project.commands infer image_path=path/to/image.jpg

# Инференс с ONNX runtime (по умолчанию)
python -m asl_project.commands infer image_path=path/to/image.jpg runtime=onnx

# Инференс с PyTorch
python -m asl_project.commands infer image_path=path/to/image.jpg runtime=torch
```

### Формат входных данных

- **Формат**: JPEG, PNG изображения
- **Размер**: Автоматически ресайзится до 200x200 (настраивается в конфиге)
- **Каналы**: RGB (3 канала)
- **Нормализация**: ImageNet статистики (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Пример данных

Примеры изображений можно найти в датасете после загрузки через DVC:

```bash
dvc pull data/ASL_Alphabet_Dataset.dvc
# Данные будут в data/ASL_Alphabet_Dataset/asl_alphabet_train/
```

Структура датасета:

```
asl_alphabet_train/
├── A/
│   ├── A1.jpg
│   ├── A2.jpg
│   └── ...
├── B/
│   └── ...
└── ...
```

### Triton Inference Server

#### Запуск Triton сервера

Triton Inference Server запускается через Docker Compose:

```bash
docker-compose up -d triton
```

#### Проверка статуса сервера

```bash
curl http://localhost:8000/v2/health/ready
```

#### Инференс через Triton

```bash
# Используйте тестовый скрипт
python scripts/test_triton_client.py path/to/image.jpg
```

Пример использования Triton client:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")
# ... подготовка входных данных ...
response = client.infer("asl_model", inputs)
output = response.as_numpy("output")
```

#### Endpoints Triton

- **HTTP**: `http://localhost:8000`
- **gRPC**: `localhost:8001`
- **Metrics**: `http://localhost:8002/metrics`

### Конфигурация инференса

Основные параметры настраиваются в `configs/inference.yaml`:

- `runtime`: "onnx" или "torch"
- `onnx_path`: путь к ONNX модели
- `checkpoint_path`: путь к PyTorch checkpoint
- `confidence_threshold`: минимальный порог уверенности
- `img_size`: размер входного изображения

## Разработка

### Запуск тестов

```bash
pytest
```

### Форматирование кода

```bash
# Автоматическое форматирование через pre-commit
pre-commit run -a
```

### Структура конфигов Hydra

Конфиги организованы иерархически:

- `configs/default.yaml` - базовые настройки
- `configs/train.yaml` - настройки обучения
- `configs/inference.yaml` - настройки инференса
- `configs/data/default.yaml` - настройки данных
- `configs/model/*.yaml` - архитектуры моделей

Переопределение параметров:

```bash
python -m asl_project.commands train model=resnet data.batch_size=64 trainer.max_epochs=50
```

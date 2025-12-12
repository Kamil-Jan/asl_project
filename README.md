# American Sign Language (ASL) Alphabet Recognition

Система распознавания букв американского языка жестов с использованием глубокого обучения. Проект реализует полный MLOps пайплайн от загрузки данных до продакшн-готового инференса с использованием современных практик машинного обучения.

## Постановка задачи

Разработать систему компьютерного зрения для классификации статических жестов американского языка жестов (ASL). Американский язык жестов используется миллионами людей, такая система может служить переводчиком или как средство для обучения.

## Формат входных и выходных данных

**Вход**: Тензор изображения формата RGB (размерности [3, H, W])

**Выход**: вектор вероятностей длиной 29 (кол-во классов) или строка - класс с наибольшей вероятностью

## Метрики

Основная метрика будет **Accuracy** (т.к. датасет сбалансирован). В качестве дополнительной метрики буду использовать **F1-Score**. Учитывая качество датасета и современные архитектуры, ожидаю получить accuracy на тесте >80%

## Валидация и тест

Будет использован **Stratified Split** исходного датасета, чтобы сохранить баланс классов во всех выборках. Для воспроизводимости результатов буду фиксировать random_seed.

Разбивать буду так:

- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%

## Датасеты

Буду использовать **ASL Alphabet**

- **Источник**: Kaggle
- **Объем**: ~1 GB
- **Кол-во семплов**: 87000
- **Кол-во классов**: 29 (A-Z, del, space, nothing)
- **Особенности**: изображения цветные (RGB), размер исходных фото 200x200 пикселей. Данные чистые, но возможна проблема переобучения на однотонный фон, что может быть проблемой при тестировании на реальных фото

## Моделирование

### Бейзлайн

В качестве baseline решения будет реализована легковесная сверточная нейросеть.

**Архитектура**: 3 сверточных слоя (Conv2d + ReLU + MaxPool) и 2 полносвязных слоя. Это позволит оценить "сложность" данных и получить начальную метрику, с которой будет сравниваться основная модель.

### Основная модель

Будет использован подход **Transfer Learning** на базе архитектуры ResNet-18 или ResNet-34 из библиотеки torchvision.

- Предобученные веса (на ImageNet) позволят модели быстрее выделить признаки формы руки.
- Последний полносвязный слой (head) будет заменен на новый с 29 выходами под текущую задачу.
- Будут применены аугментации (повороты, изменение яркости) для повышения устойчивости модели.

## Внедрение

Модель будет упакована в виде python-пакета, который будет принимать путь к файлу изображения и возвращать предсказанную букву. Если хватит времени и сил, попробую брать картинку напрямую с веб-камеры и транслировать в реальном времени.

---

## Setup

Данный раздел описывает процедуру настройки окружения проекта для нового члена команды. После выполнения всех шагов вы сможете продолжить разработку, запустить обучение и выполнить предсказание модели.

### Требования

- Python >= 3.10
- Poetry (для управления зависимостями)
- Git
- DVC (для управления данными)
- Docker и Docker Compose (для MLflow)

### Установка окружения

1. **Клонируйте репозиторий:**

   ```bash
   git clone <repository-url>
   cd mlops
   ```

2. **Установите Poetry (если еще не установлен):**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Или через pip:

   ```bash
   pip install poetry
   ```

3. **Установите зависимости проекта:**

   ```bash
   poetry install
   ```

   Это установит все необходимые зависимости, включая:
   - PyTorch и PyTorch Lightning
   - Hydra для управления конфигурациями
   - MLflow для отслеживания экспериментов
   - DVC для управления данными
   - ONNX и ONNX Runtime для экспорта и инференса
   - И другие зависимости из `pyproject.toml`

4. **Активируйте окружение:**

   ```bash
   poetry shell
   ```

   Или используйте команды через `poetry run`:

   ```bash
   poetry run python -m asl_project.commands train
   ```

5. **Настройте pre-commit hooks (опционально, но рекомендуется):**

   ```bash
   pre-commit install
   ```

   Это настроит автоматическую проверку кода при коммитах (форматирование, линтинг).

6. **Проверьте установку:**

   ```bash
   pre-commit run -a
   ```

7. **Настройте DVC (если используется удаленное хранилище):**

   Если данные хранятся в удаленном хранилище (S3, GCS и т.д.), настройте DVC:

   ```bash
   dvc remote add -d <remote-name> <remote-url>
   ```

### Проверка готовности окружения

После установки проверьте, что все работает:

```bash
# Проверка Python окружения
poetry run python --version

# Проверка доступности модулей
poetry run python -c "import torch; import pytorch_lightning; print('OK')"

# Проверка DVC
dvc --version
```

Теперь окружение готово для работы с проектом!

## Train

Данный раздел описывает процесс обучения модели, включая загрузку данных, предобработку и запуск обучения для различных вариантов моделей.

### Загрузка данных

Данные автоматически загружаются через DVC при первом запуске обучения. Если данные отсутствуют локально, они будут скачаны из удаленного хранилища (S3) или напрямую из открытых источников (если настроено).

**Ручная загрузка данных:**

```bash
# Загрузка данных через DVC
dvc pull data/ASL_Alphabet_Dataset.dvc
```

После загрузки данные будут находиться в `data/ASL_Alphabet_Dataset/`:

```
data/ASL_Alphabet_Dataset/
├── asl_alphabet_train/    # Обучающая выборка
│   ├── A/
│   ├── B/
│   └── ...
└── asl_alphabet_test/     # Тестовая выборка
```

### Предобработка данных

Предобработка выполняется автоматически при загрузке данных в DataModule:

- **Разделение на train/val/test**: Stratified split с сохранением баланса классов
- **Аугментации для обучения**:
  - Resize до 200x200
  - Повороты (до 20 градусов)
  - Горизонтальное отражение
  - Изменение яркости и контраста
  - Гауссов шум и размытие
  - Нормализация ImageNet статистиками
- **Препроцессинг для валидации/теста**:
  - Resize до 200x200
  - Нормализация ImageNet статистиками

### Запуск обучения

Обучение запускается через единую точку входа `commands.py` с использованием CLI интерфейса на базе Fire.

#### Обучение Baseline модели

```bash
# Обучение Baseline CNN с дефолтными параметрами
python -m asl_project.commands train model=baseline

# Обучение с кастомными параметрами
python -m asl_project.commands train \
    model=baseline \
    trainer.max_epochs=50 \
    data.batch_size=32 \
    model.lr=0.001
```

#### Обучение ResNet модели (основная модель)

```bash
# Обучение ResNet-18 с дефолтными параметрами
python -m asl_project.commands train model=resnet

# Обучение ResNet-18 с кастомными параметрами
python -m asl_project.commands train \
    model=resnet \
    data.batch_size=64 \
    trainer.max_epochs=10 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    model.lr=0.0003

# Обучение ResNet-34
python -m asl_project.commands train \
    model=resnet \
    model.backbone_name=resnet34
```

#### Переопределение параметров через CLI

Все параметры можно переопределять через командную строку:

```bash
python -m asl_project.commands train \
    model=resnet \
    data.batch_size=64 \
    data.val_split=0.15 \
    data.test_split=0.05 \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    logger.experiment_name=my_experiment \
    seed=42
```

### Мониторинг обучения

Во время обучения метрики автоматически логируются в MLflow:

- **Метрики**: train_loss, train_acc, val_loss, val_acc, val_f1
- **Гиперпараметры**: все параметры модели и обучения
- **Артефакты**:
  - Чекпоинты моделей
  - Графики метрик (loss_curve.png, acc_curve.png)
  - Матрица ошибок (confusion_matrix.png)
  - ONNX модель (если включен auto_export_onnx)

**Запуск MLflow UI:**

```bash
# Запуск PostgreSQL для MLflow (если используется)
docker-compose up -d postgres

# Запуск MLflow UI
mlflow ui --backend-store-uri postgresql://mlflow:mlflow@localhost:5433/mlflow --port 8080

# Или с локальным хранилищем
mlflow ui --port 8080
```

Откройте http://localhost:8080 для просмотра экспериментов.

### Результаты обучения

После обучения:

- **Чекпоинты** сохраняются в `models/checkpoints/`
- **Лучшая модель** копируется в `models/checkpoints/best.ckpt`
- **Графики** сохраняются в `plots/`
- **ONNX модель** автоматически экспортируется в `models/model.onnx` (если включено)

## Production preparation

Данный раздел описывает шаги подготовки обученной модели к работе в продакшене, включая экспорт в ONNX и комплектацию поставки.

### Экспорт модели в ONNX

После обучения модель автоматически экспортируется в ONNX формат (если `auto_export_onnx: true` в конфиге). Также можно экспортировать вручную:

```bash
# Экспорт в ONNX из чекпоинта
python -m asl_project.commands export \
    checkpoint_path=models/checkpoints/best.ckpt \
    onnx_path=models/model.onnx
```

**Параметры экспорта:**

- **opset_version**: 18 (совместимость с ONNX Runtime)
- **dynamic_axes**: поддержка динамического batch size
- **input_names**: ["input"]
- **output_names**: ["output"]

ONNX модель будет сохранена в `models/model.onnx` (или путь из конфига).

## Infer

Данный раздел описывает процесс запуска обученной модели на новых данных для получения предсказаний.

### Инференс на изображении

Инференс запускается через CLI интерфейс:

```bash
# Инференс на одном изображении с ONNX (по умолчанию)
python -m asl_project.commands infer image_path=path/to/image.jpg

# Инференс с указанием полного пути
python -m asl_project.commands infer image_path=/absolute/path/to/image.jpg

# Инференс с PyTorch (если ONNX недоступен)
python -m asl_project.commands infer \
    image_path=path/to/image.jpg \
    runtime=torch \
    checkpoint_path=models/checkpoints/best.ckpt
```

### Инференс через веб-камеру

Для работы в реальном времени с веб-камерой:

```bash
# Запуск инференса в реальном времени
python -m asl_project.commands webcam

# С кастомными параметрами
python -m asl_project.commands webcam \
    runtime=onnx \
    confidence_threshold=0.5 \
    onnx_path=models/model.onnx
```

Веб-камера откроет окно с предпросмотром, где будет выделена область для распознавания (400x400 пикселей в центре кадра).

### Формат входных данных

**Требования к входным данным:**

- **Формат файлов**: JPEG, PNG
- **Размер**: Автоматически ресайзится до 200x200 (настраивается в конфиге через `img_size`)
- **Каналы**: RGB (3 канала)
- **Нормализация**: ImageNet статистики (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Препроцессинг выполняется автоматически:**

1. Загрузка изображения
2. Конвертация BGR → RGB
3. Resize до 200x200
4. Нормализация ImageNet статистиками
5. Преобразование в тензор [1, 3, 200, 200]

### Пример данных

Примеры изображений можно найти в датасете после загрузки через DVC:

```bash
# Загрузка данных
dvc pull data/ASL_Alphabet_Dataset.dvc

# Примеры изображений находятся в:
# data/ASL_Alphabet_Dataset/asl_alphabet_train/A/A1.jpg
# data/ASL_Alphabet_Dataset/asl_alphabet_train/B/B1.jpg
# и т.д.

# Тестовые изображения:
# data/ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg
# data/ASL_Alphabet_Dataset/asl_alphabet_test/B_test.jpg
```

**Структура датасета:**

```
asl_alphabet_train/
├── A/
│   ├── A1.jpg
│   ├── A2.jpg
│   └── ...
├── B/
│   ├── B1.jpg
│   └── ...
├── ...
├── del/
├── nothing/
└── space/
```

**Пример запуска инференса на тестовом изображении:**

```bash
python -m asl_project.commands infer \
    image_path=data/ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg
```

**Ожидаемый вывод:**

```
Result: A (confidence: 0.95)
```

### Конфигурация инференса

Основные параметры настраиваются в `configs/inference.yaml` или переопределяются через CLI:

- `runtime`: "onnx" или "torch" - выбор рантайма для инференса
- `onnx_path`: путь к ONNX модели (по умолчанию `models/model.onnx`)
- `checkpoint_path`: путь к PyTorch checkpoint (для runtime=torch)
- `confidence_threshold`: минимальный порог уверенности (по умолчанию 0.3)
- `img_size`: размер входного изображения [200, 200]
- `class_names`: список из 29 классов ASL алфавита

**Пример переопределения параметров:**

```bash
python -m asl_project.commands infer \
    image_path=path/to/image.jpg \
    runtime=onnx \
    onnx_path=models/model.onnx \
    confidence_threshold=0.5 \
    img_size=[224,224]
```

# Model Pipeline

Папка `model_pipeline/` содержит код обучения модели кредитного скоринга, данные, тесты и артефакты запуска.

## Запуск

Из корня репозитория:

```bash
bash model_pipeline/run_pipeline.sh
```

Полный прогон с расширенным подбором гиперпараметров:

```bash
bash model_pipeline/run_pipeline.sh --full
```

Если данные лежат вне репозитория, можно передать путь к директории `GiveMeSomeCredit`:

```bash
bash model_pipeline/run_pipeline.sh --data-dir /path/to/GiveMeSomeCredit
```

Полный прогон с внешней директорией данных:

```bash
bash model_pipeline/run_pipeline.sh --full --data-dir /path/to/GiveMeSomeCredit
```

Ожидается, что в директории с данными лежат файлы:

- `cs-training.csv`
- `cs-test.csv`

По умолчанию скрипт ищет их в `model_pipeline/GiveMeSomeCredit/`.
Если файлов нет, пайплайн выведет понятное сообщение с ссылкой на соревнование Kaggle:
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data).

## Структура

```text
model_pipeline/
├── GiveMeSomeCredit/
├── artifacts/
├── outputs/
├── src/
├── templates/
├── tests/
├── README.md
├── requirements.txt
└── run_pipeline.sh
```

- `GiveMeSomeCredit/` — исходные данные для обучения и инференса.
- `artifacts/` — результаты работы пайплайна: модель, метрики, отчеты, графики и предсказания.
- `outputs/` — дополнительные выходные материалы, связанные с генерацией презентаций.
- `src/` — исходный код feature engineering, обучения, оценки и отчетности.
- `templates/` — шаблоны для генерации отчетов.
- `tests/` — тесты для основных модулей пайплайна.
- `requirements.txt` — зависимости Python.
- `run_pipeline.sh` — основной скрипт запуска обучения.

## Ключевые Файлы

- Код обучения: [src/train_pipeline.py](/Users/igor/Repositories/fintech_ai_light/model_pipeline/src/train_pipeline.py)
- Модуль моделирования: [src/modeling.py](/Users/igor/Repositories/fintech_ai_light/model_pipeline/src/modeling.py)
- Скрипт запуска: [run_pipeline.sh](/Users/igor/Repositories/fintech_ai_light/model_pipeline/run_pipeline.sh)
- Артефакты после запуска: [artifacts](/Users/igor/Repositories/fintech_ai_light/model_pipeline/artifacts)

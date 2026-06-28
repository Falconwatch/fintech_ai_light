# Fintech AI Light

Репозиторий содержит:

- `model_pipeline/` — пайплайн обучения модели кредитного скоринга;
- `task_done/` — финальные материалы по заданию.

## Быстрый старт

Из корня репозитория:

```bash
bash model_pipeline/run_pipeline.sh
```

Полный прогон с увеличенным числом `Optuna`-триалов:

```bash
bash model_pipeline/run_pipeline.sh --full
```

Если данные лежат вне стандартной директории, можно передать путь явно:

```bash
bash model_pipeline/run_pipeline.sh --data-dir /path/to/GiveMeSomeCredit
```

Также поддерживается полный режим с внешней директорией данных:

```bash
bash model_pipeline/run_pipeline.sh --full --data-dir /path/to/GiveMeSomeCredit
```

## Где могут лежать данные

Пайплайн умеет искать данные в нескольких местах:

- `model_pipeline/GiveMeSomeCredit/`
- `GiveMeSomeCredit/` в корне репозитория
- во внешней директории, переданной через `--data-dir`

Ожидаются файлы:

- `cs-training.csv`
- `cs-test.csv`

Если данных нет, скрипт выведет понятное сообщение с ссылкой на Kaggle:
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data)

## Структура

```text
.
├── model_pipeline/
├── task_done/
└── README.md
```

- `model_pipeline/` — код пайплайна, шаблоны, тесты, зависимости, скрипт запуска и локальные артефакты после выполнения.
- `task_done/` — финальные материалы по заданию и исходный документ с требованиями.

## Где смотреть подробнее

- Пайплайн: [model_pipeline/README.md](/Users/igor/Repositories/fintech_ai_light/model_pipeline/README.md)
- Материалы задания: [task_done/README.md](/Users/igor/Repositories/fintech_ai_light/task_done/README.md)

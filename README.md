# Fintech AI Light

Репозиторий содержит пайплайн обучения модели кредитного скоринга и материалы по заданию.

## Быстрый старт

Скачать репозиторий и запустить обучение модели:

```bash
git clone <repo_url>
cd fintech_ai_light
bash model_pipeline/run_pipeline.sh
```

Если нужен более длинный прогон с большим числом `Optuna`-триалов:

```bash
bash model_pipeline/run_pipeline.sh --full
```

Если данные лежат вне репозитория, можно передать путь к папке `GiveMeSomeCredit` явно:

```bash
bash model_pipeline/run_pipeline.sh --data-dir /path/to/GiveMeSomeCredit
```

Что делает скрипт `model_pipeline/run_pipeline.sh`:

- использует виртуальное окружение в `model_pipeline/.venv`;
- при необходимости обновляет `pip` и ставит зависимости из `model_pipeline/requirements.txt`;
- запускает обучение модели на данных из `model_pipeline/GiveMeSomeCredit/` или из внешней папки, переданной через `--data-dir`;
- сохраняет модель, метрики, графики и markdown-отчет в `model_pipeline/artifacts/`.

Подробности по ML-части и содержимому папки `model_pipeline/` вынесены в [model_pipeline/README.md](/Users/igor/Repositories/fintech_ai_light/model_pipeline/README.md).
Описание материалов по заданию вынесено в `task_done/README.md`.

## Структура Корня

```text
.
├── task_done/
├── model_pipeline/
└── README.md
```

- `task_done/` — все материалы по заданию: текстовые артефакты, презентации, план видео, архитектура и вспомогательные документы.
  Подробное описание лежит в `task_done/README.md`.
- `model_pipeline/` — код пайплайна, данные, шаблоны, тесты, скрипт запуска и артефакты обучения модели.
  Подробное описание лежит в `model_pipeline/README.md`.
- `README.md` — краткая инструкция по запуску и обзор корневой структуры.

## Что Где Лежит

- Основной код: [model_pipeline/src](/Users/igor/Repositories/fintech_ai_light/model_pipeline/src)
- README пайплайна: [model_pipeline/README.md](/Users/igor/Repositories/fintech_ai_light/model_pipeline/README.md)
- README материалов: [task_done/README.md](/Users/igor/Repositories/fintech_ai_light/task_done/README.md)
- Скрипт запуска: [model_pipeline/run_pipeline.sh](/Users/igor/Repositories/fintech_ai_light/model_pipeline/run_pipeline.sh)
- Данные: [model_pipeline/GiveMeSomeCredit](/Users/igor/Repositories/fintech_ai_light/model_pipeline/GiveMeSomeCredit)
- Результаты обучения: [model_pipeline/artifacts](/Users/igor/Repositories/fintech_ai_light/model_pipeline/artifacts)
- Материалы по заданию: [task_done](/Users/igor/Repositories/fintech_ai_light/task_done)

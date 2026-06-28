#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-5}"
DATA_DIR="${ROOT_DIR}/GiveMeSomeCredit"
export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      OPTUNA_TRIALS=25
      shift
      ;;
    --data-dir)
      if [[ $# -lt 2 ]]; then
        echo "--data-dir requires a path argument"
        exit 1
      fi
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./run_pipeline.sh [--full] [--data-dir /path/to/GiveMeSomeCredit]"
      exit 1
      ;;
  esac
done

TRAIN_PATH="${DATA_DIR}/cs-training.csv"
TEST_PATH="${DATA_DIR}/cs-test.csv"

log() {
  printf '[%s] [run_pipeline] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

error() {
  printf '\033[31m%s\033[0m\n' "$1"
}

log "Старт пайплайна"
log "Корневая директория: ${ROOT_DIR}"
log "PYTHONPATH: ${PYTHONPATH}"
log "Директория с данными: ${DATA_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Создаю виртуальное окружение в ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  log "Виртуальное окружение уже существует"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  log "Не найден исполняемый файл ${VENV_PYTHON}"
  exit 1
fi

log "Использую интерпретатор виртуального окружения: ${VENV_PYTHON}"
log "Обновляю pip"
"${VENV_PYTHON}" -m pip install --upgrade pip >/dev/null 2>&1
log "Устанавливаю зависимости из requirements.txt"
"${VENV_PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.txt" >/dev/null 2>&1
log "Зависимости установлены"

log "Запускаю обучение и генерацию отчета"
if [[ ! -f "${TRAIN_PATH}" || ! -f "${TEST_PATH}" ]]; then
  error "Упс, кажется вы не скачали данные для обучения. Для работы пайплайна скачайте данные с https://www.kaggle.com/c/GiveMeSomeCredit/data и разместите их либо в ${ROOT_DIR}/GiveMeSomeCredit/ так, чтобы там лежали файлы cs-training.csv и cs-test.csv, либо передайте внешний путь через --data-dir /path/to/GiveMeSomeCredit"
  exit 1
fi

"${VENV_PYTHON}" "${ROOT_DIR}/src/train_pipeline.py" \
  --train-path "${TRAIN_PATH}" \
  --test-path "${TEST_PATH}" \
  --output-dir "${ROOT_DIR}/artifacts" \
  --optuna-trials "${OPTUNA_TRIALS}"
log "Пайплайн завершен"

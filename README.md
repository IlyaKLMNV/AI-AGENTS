# AI Agents Workspace

Набор утилит и промптов для тестирования рекрутмент-агентов. В репозитории есть полный пайплайн проверки нескольких модулей и отдельный скрипт для точечной проверки промпта `screening_assistant`.

## Структура
- `app/runner.py` — основной тестовый пайплайн (генерация фикстур, интеграционный прогон всех модулей).
- `app/screening_scenarios_runner.py` — проверка одного промпта `screening_assistant` на наборе поведенческих сценариев.
- `app/screening_guardrails_runner.py` — guardrails-проверки ответов рекрутера (self_answer, repeated_questions, premature_end).
- `app/message_classifier_runner.py` — тестирование `message_classifier` по классам.
- `app/telegram_generator_runner.py` — тест первого касания `telegramMessageGenerator` по CDM.
- `tests/tools/model.yaml` — конфигурация с `prompt_id`/`prompt_version` для всех промптов (message_classifier, screening_assistant, screening_autofill, verdict_classifier, candidate_simulator профили).
- `tests/fixtures/cdm/` — CDM-фикстуры с вакансиями (генерируются `gen-fixtures`).
- `tests/fixtures/screening_scenarios.csv` — CSV со сценариями для точечной проверки `screening_assistant`.
- `tests/reports/runs/` — отчёты основного пайплайна.
- `tests/reports/screening_scenarios/` — отчёты по скрипту `screening_scenarios_runner.py`.
- `tests/reports/screening_guardrails/` — отчёты `screening_guardrails_runner.py`.
- `tests/reports/message_classifier/` — отчёты `message_classifier_runner.py`.
- `tests/reports/telegram_generator/` — отчёты `telegram_generator_runner.py`.
- `telegramMessageGenerator-main/` — опциональный генератор первого сообщения в Telegram (используется, если установлен и импортируется).

## Раннеры
- `app/runner.py` — основной тестовый пайплайн. Запуск: `python -m app.runner gen-fixtures`; `python -m app.runner unit --limit 5 --candidate-profiles difficult ideal`. Отчёты: `tests/reports/runs/`.
- `app/screening_scenarios_runner.py` — проверка `screening_assistant` на сценариях. Запуск: `python -m app.screening_scenarios_runner --csv-path tests/fixtures/screening_scenarios.csv --messages-per-scenario 3 --max-scenarios 20`. Отчёты: `tests/reports/screening_scenarios/`.
- `app/screening_guardrails_runner.py` — guardrails-проверка ответов рекрутера. Запуск: `python -m app.screening_guardrails_runner --conversations 20 --turns-per-conversation 4 --report-mode compact`. Отчёты: `tests/reports/screening_guardrails/`.
- `app/message_classifier_runner.py` — тест message_classifier по классам. Запуск: `python -m app.message_classifier_runner --n-per-class 3 --seed 42`. Отчёты: `tests/reports/message_classifier/`.
- `app/telegram_generator_runner.py` — тест первого касания `telegramMessageGenerator` по CDM. Запуск: `python -m app.telegram_generator_runner --limit 5 --require-question`. Отчёты: `tests/reports/telegram_generator/`. Нужен `OPENAI_API_KEY`; `FIRST_TOUCH_PROMPT_ID` берётся из env или `tests/tools/model.yaml:first_touch`.

## Подготовка окружения
```bash
# WSL (Ubuntu)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Создайте `.env` (можно скопировать из `.env.example`) и пропишите ключ:
```bash
OPENAI_API_KEY=sk-...
```

Пример запуска в WSL:
```bash
python3 -m venv .venv
source .venv/bin/activate
export OPENAI_API_KEY='sk-...'
python -m app.telegram_generator_runner --limit 5 --require-question
```

Заполните `tests/tools/model.yaml`: задайте `prompt_id`/`prompt_version` для всех компонентов и профили в блоке `candidate_simulator` (ключи совпадают с именами профилей, которые передаются в `--candidate-profiles`).

## Основной тестовый пайплайн (`app/runner.py`)

### Что делает
- Берёт CDM-фикстуры из `tests/fixtures/cdm/` (каждая описывает вакансию и кандидата).
- Для каждого профиля из `candidate_simulator` создаёт диалог: строит стартовое сообщение (через `telegramMessageGenerator`, шаблон из CDM или fallback), далее симулирует кандидата, прогоняет диалог через `screening_assistant`, классификатор сообщений, классификатор вердикта и `screening_autofill` для passed-кейсов.
- Сохраняет помодульные метрики, usage по токенам и отчёты в `tests/reports/runs/<run_id>/`.

### Что нужно перед запуском
- `tests/tools/model.yaml` должен содержать валидные prompt_id/prompt_version для всех компонентов и профилей.
- В `tests/fixtures/cdm/` должны лежать CDM-файлы (сгенерируйте через `gen-fixtures`, если пусто).
- Переменная окружения `OPENAI_API_KEY` должна быть доступна процессу.
- Если нужен генератор Telegram-сообщений, убедитесь, что `telegramMessageGenerator-main` импортируется; иначе будет использован шаблон из CDM или встроенный fallback.

### Запуск
Генерация фикстур (10 CDM по умолчанию):
```bash
python -m app.runner gen-fixtures
```

Интеграционный прогон:
```bash
python -m app.runner unit --limit 5 --candidate-profiles difficult ideal
```
Аргументы:
- `--limit` — сколько CDM из `tests/fixtures/cdm/` брать в прогон (по умолчанию 5).
- `--candidate-profiles` — какие профили из `candidate_simulator` использовать (по умолчанию все профили из `model.yaml`).

### Результаты
- Сводка: `tests/reports/runs/<run_id>/report-<run_id>.json`
- Отчёты по отдельным диалогам: `tests/reports/runs/<run_id>/dialogs/*.json`
В отчётах фиксируются успех/провалы модулей, вердикт, автозаполнение, usage и длительность.

## Проверка промпта `screening_assistant` по сценариям (`app/screening_scenarios_runner.py`)

### Назначение
Гоняет один промпт `screening_assistant` по набору поведенческих сценариев из CSV: генерирует сообщения кандидата, получает ответ ассистента и жёстко сравнивает его с ожидаемым поведением.

### Что нужно перед запуском
- `OPENAI_API_KEY` в окружении.
- В `tests/tools/model.yaml` в блоке `screening_assistant` должны быть заполнены `prompt_id` и (опционально) `prompt_version` нужного промпта.
- CSV с сценариями (по умолчанию `tests/fixtures/screening_scenarios.csv`). Ожидаются колонки: `Название сценария`, `Краткое описание сценария`, поле с ожидаемым поведением (несколько вариантов названий) и поле с примерами диалогов.

### Запуск
```bash
python -m app.screening_scenarios_runner \
  --csv-path tests/fixtures/screening_scenarios.csv \
  --messages-per-scenario 3 \
  --max-scenarios 20
```

Параметры:
- `--csv-path` — путь к CSV со сценариями (по умолчанию `tests/fixtures/screening_scenarios.csv`).
- `--messages-per-scenario` — сколько реплик кандидата генерировать на сценарий (по умолчанию 3).
- `--max-scenarios` — ограничение числа сценариев для быстрого прогона (по умолчанию нет, берутся все).

### Как проходит тестирование
1. Читаем сценарии из CSV и (опционально) обрезаем по `--max-scenarios`.
2. Генерируем кандидатовские сообщения под каждый сценарий (`gpt-4.1-mini`).
3. Прокидываем каждое сообщение в `screening_assistant` (prompt_id из `model.yaml`), собираем ответы.
4. Оцениваем каждый ответ моделью `gpt-4.1` на соответствие ожидаемому поведению из CSV.
5. Формируем отчёт `tests/reports/screening_scenarios/screening_scenarios_report_<timestamp>.json` с токенами, пройденными/заваленными шагами и использованием моделей.

Запуск на Windows и Linux одинаковый (главное, чтобы активированное venv и переменная `OPENAI_API_KEY` были доступны).

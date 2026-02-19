# AI Agents Workspace

Набор раннеров для тестирования рекрутмент-промптов. Включает полный интеграционный прогон и отдельные проверки для `screening_assistant`, `message_classifier`, `screening_autofill`, `verdict_classifier` и генератора первого касания (Telegram).

## Структура проекта
- `app/` — CLI-раннеры.
- `adapters/` — преобразования CDM -> input-форматы промптов.
- `cdm/` — схема CDM и примерные данные.
- `messageLabelGenerator/` — обвязка промпта `message_classifier`.
- `screeningAssistant/` — обвязка промпта `screening_assistant`.
- `screening_autofill/` — обвязка промпта `screening_autofill`.
- `verdict_classifier/` — обвязка промпта `verdict_classifier`.
- `telegramMessageGenerator-main/` — опциональный генератор первого сообщения в Telegram.
- `tests/fixtures/` — фикстуры (CDM и сценарии).
- `tests/tools/` — `model.yaml` и скрипты генерации фикстур.
- `tests/reports/` — отчеты раннеров (`runs`, `screening_scenarios`, `screening_guardrails`, `message_classifier`, `telegram_generator`, `screening_autofill`, `verdict_classifier`).

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

Создайте `.env` (можно скопировать из `.env.example`) и задайте ключ:
```bash
OPENAI_API_KEY=sk-...
```

Все раннеры, которые используют LLM, требуют `OPENAI_API_KEY`. Исключение: `gen-fixtures`.

## Конфигурация промптов и профилей
`tests/tools/model.yaml` хранит `prompt_id`/`prompt_version` для:
- `first_touch`
- `message_classifier`
- `screening_assistant`
- `screening_autofill`
- `verdict_classifier`
- `candidate_simulator` (профили кандидатов для `app/runner.py`)

Переменные окружения для переопределения:
- `FIRST_TOUCH_PROMPT_ID` — для `app/telegram_generator_runner.py` и генератора первого касания в `app/runner.py`.
- `SCREENING_AUTOFILL_PROMPT_ID`, `SCREENING_AUTOFILL_PROMPT_VERSION` — для `app/screening_autofill_runner.py`.
- `VERDICT_CLASSIFIER_PROMPT_ID`, `VERDICT_CLASSIFIER_PROMPT_VERSION` — для `app/verdict_classifier_runner.py`.

## Фикстуры и данные
- `tests/fixtures/cdm/` — CDM-фикстуры вакансий (генерируются `python -m app.runner gen-fixtures`).
- `tests/fixtures/screening_scenarios.csv` — сценарии для проверки `screening_assistant`.
- `cdm/schema.json` — схема CDM.

## Раннеры (`app/`)

### `app/runner.py` — интеграционный прогон пайплайна
Как работает:
- Берет CDM-фикстуры и прогоняет их по всем профилям из `candidate_simulator`.
- Стартовое сообщение генерируется через `telegramMessageGenerator` (если доступен), иначе берется шаблон из CDM или fallback.
- Диалог проходит через `screening_assistant`, `message_classifier`, `verdict_classifier`, `screening_autofill`, собираются метрики и usage.

Запуск:
```bash
python -m app.runner gen-fixtures
python -m app.runner unit --limit 5 --candidate-profiles difficult ideal
```

Параметры:
- `gen-fixtures` — генерирует `tests/fixtures/cdm/cdm_*.json` (ключ не нужен).
- `unit --limit` — сколько CDM брать в прогон (по умолчанию 5).
- `unit --candidate-profiles` — список профилей из `candidate_simulator` (по умолчанию все).

Отчеты:
- `tests/reports/runs/<run_id>/report-<run_id>.json`
- `tests/reports/runs/<run_id>/dialogs/*.json`

### `app/screening_scenarios_runner.py` — сценарии для `screening_assistant`
Как работает:
- Читает CSV со сценариями, группирует одиночные и цепочные сценарии.
- Загружает вакансии из `tests/fixtures/cdm/*.json` и передает в prompt контекст вакансии (имя рекрутера/кандидата, роль, компания, обязанности, формат, описание, ссылка, salary, вопросы).
- CSV ожидает колонки: `Название сценария`, `Краткое описание сценария`, поле с ожидаемым поведением и поле с примерами диалогов.
- Генерирует реплики кандидата и получает ответы `screening_assistant`.
- Для сценариев `23/24` (скрытый поиск) принудительно подставляет `Компания: СКРЫТО` в передаваемый контекст.
- Оценивает соответствие ожидаемому поведению через LLM-судью.

Запуск:
```bash
python -m app.screening_scenarios_runner \
  --csv-path tests/fixtures/screening_scenarios.csv \
  --cdm-dir tests/fixtures/cdm \
  --messages-per-scenario 3 \
  --max-scenarios 20
```

Параметры:
- `--csv-path` — путь к CSV со сценариями.
- `--cdm-dir` — директория с CDM-фикстурами вакансий.
- `--messages-per-scenario` — SINGLE: сообщений на сценарий; CHAIN: прогонов диалога на цепочку.
- `--max-scenarios` — лимит сценариев для быстрого прогона.

Отчеты:
- `tests/reports/screening_scenarios/screening_scenarios_report_<timestamp>.json`

### `app/screening_guardrails_runner.py` — guardrails-проверки
Как работает:
- Генерирует многоходовые диалоги кандидата.
- Прогоняет их через `screening_assistant`.
- Проверяет ответы рекрутера на `self_answer`, `repeated_questions`, `premature_end_after_questions` (LLM + эвристики).

Запуск:
```bash
python -m app.screening_guardrails_runner --conversations 20 --turns-per-conversation 4 --report-mode compact
```

Параметры:
- `--conversations` — число диалогов (по умолчанию 50).
- `--turns-per-conversation` — число реплик кандидата в диалоге (по умолчанию 4).
- `--report-mode` — `compact`, `full` или `both`.

Отчеты:
- `tests/reports/screening_guardrails/screening_guardrails_<run_id>_compact.json`
- `tests/reports/screening_guardrails/screening_guardrails_<run_id>.json` (если `full`/`both`)

### `app/message_classifier_runner.py` — тест `message_classifier`
Как работает:
- Генерирует сообщения кандидата по классам (`reason_farewell`, `no_reason`, `acceptance`, `human_needed`).
- Опционально фильтрует кейсы через LLM-судью.
- Классифицирует и считает accuracy/матрицу ошибок.

Запуск:
```bash
python -m app.message_classifier_runner --n-per-class 3 --seed 42
```

Параметры:
- `--n-per-class` — сколько кейсов на класс.
- `--seed` — сид для генерации/перемешивания.
- `--no-gen` — не генерировать LLM-кейсы, использовать только fallback-пул.
- `--judge` — включить LLM-судью для фильтрации спорных кейсов.

Отчеты:
- `tests/reports/message_classifier/message_classifier_report_<run_id>.json`

### `app/telegram_generator_runner.py` — тест первого касания (Telegram)
Как работает:
- Строит `InputForm` из CDM и генерирует сообщение через `telegramMessageGenerator`.
- Проверяет наличие фактов и галлюцинаций с помощью LLM-оценщика.
- Опционально требует вопрос в сообщении.
- Требует `telegramMessageGenerator-main` (если модуль не импортируется, раннер завершится с ошибкой).

Запуск:
```bash
python -m app.telegram_generator_runner --limit 5 --require-question
```

Параметры:
- `--limit` — сколько CDM-фикстур взять.
- `--cdm-dir`, `--out-dir` — пути к фикстурам и отчетам.
- `--eval-model` — модель-оценщик (по умолчанию `gpt-4.1-mini`).
- `--require-question` / `--no-require-question` — требовать ли вопрос.
- `--include-salary` — учитывать salary_range среди обязательных фактов.
- `--hide-company` — скрывать компанию во всех кейсах.
- `--hide-company-ratio` — скрывать компанию в доле кейсов (0..1).
- `--seed` — сид для `--hide-company-ratio`.

Отчеты:
- `tests/reports/telegram_generator/telegram_generator_report_<run_id>.json`

### `app/screening_autofill_runner.py` — тест `screening_autofill`
Как работает:
- Генерирует диалоги на основе CDM (несколько вариантов на вакансию).
- Прогоняет диалоги через `screening_autofill` и парсит JSON.
- Валидирует схему и семантическую согласованность ответов.
- `prompt_id` берется из CLI/env/model.yaml, без него запуск невозможен.

Запуск:
```bash
python -m app.screening_autofill_runner --cdm-count 5 --variants-per-cdm 3
```

Параметры:
- `--cdm-dir` — путь к CDM.
- `--cdm-count` — сколько CDM взять (по умолчанию все).
- `--variants-per-cdm` — число диалогов на CDM.
- `--noise-level` — 0..2, уровень шума в ответах.
- `--allow-two-questions` — разрешить два вопроса в реплике рекрутера.
- `--flatten-like-prod` — преобразовать диалог в одну строку (как в проде).
- `--seed` — сид для вариативности.
- `--dialogue-gen-model` — модель генерации диалогов.
- `--prompt-id`, `--prompt-version` — переопределить промпт.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/screening_autofill/screening_autofill_report_<run_id>.json`

### `app/verdict_classifier_runner.py` — тест `verdict_classifier`
Как работает:
- Генерирует диалоги с целевыми вердиктами `passed/failed/deadlock`.
- Прогоняет их через `verdict_classifier` и считает точность.
- Управляет набором сценариев через режимы `random`/`cycle`.
- `prompt_id` берется из CLI/env/model.yaml, без него запуск невозможен.

Запуск:
```bash
python -m app.verdict_classifier_runner --dialogs-per-verdict 5
```

Параметры:
- `--cdm-dir` — путь к CDM.
- `--cdm-count` — сколько CDM взять (по умолчанию все).
- `--dialogs-per-verdict` — число диалогов на каждый вердикт (обязательный).
- `--noise-level` — 0..2, уровень шума.
- `--seed` — сид (переопределяет сид из `model.yaml`).
- `--dialogue-gen-model` — модель генерации диалогов.
- `--prompt-id`, `--prompt-version` — переопределить промпт.
- `--scenario-mode` — `random` или `cycle`.
- `--scenario-count-per-verdict` — ограничить число сценариев на вердикт.
- `--max-attempts-multiplier` — лимит попыток генерации.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/verdict_classifier/verdict_classifier_report_<run_id>.json`

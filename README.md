# AI Agents Workspace

Набор раннеров для тестирования рекрутмент-промптов. Включает полный интеграционный прогон и отдельные проверки для `screening_assistant`, `message_classifier`, `screening_autofill`, `verdict_classifier`, `extractor_agent`, `sourcing_assistant` и генератора первого касания (Telegram).

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
- `tests/reports/` — отчеты раннеров (`runs`, `screening_scenarios`, `screening_guardrails`, `message_classifier`, `telegram_generator`, `screening_autofill`, `verdict_classifier`, `extractor_agent_full`).

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
- `extractor_agent`
- `sourcing_assistant`
- `candidate_simulator` (профили кандидатов для `app/runner.py`)

Переменные окружения для переопределения:
- `FIRST_TOUCH_PROMPT_ID` — для `app/telegram_generator_runner.py` и генератора первого касания в `app/runner.py`.
- `SCREENING_AUTOFILL_PROMPT_ID`, `SCREENING_AUTOFILL_PROMPT_VERSION` — для `app/screening_autofill_runner.py`.
- `VERDICT_CLASSIFIER_PROMPT_ID`, `VERDICT_CLASSIFIER_PROMPT_VERSION` — для `app/verdict_classifier_runner.py`.
- `EXTRACTOR_AGENT_PROMPT_ID`, `EXTRACTOR_AGENT_PROMPT_VERSION` — для `app/extractor_agent_runner.py`.
- `SOURCING_ASSISTANT_PROMPT_ID`, `SOURCING_ASSISTANT_PROMPT_VERSION` — для `app/sourcing_assistant_runner.py`.

## Фикстуры и данные
- `tests/fixtures/cdm/` — CDM-фикстуры вакансий (генерируются `python -m app.runner gen-fixtures`).
  Для `sourcing_assistant_runner.py` в `vacancy` дополнительно используются:
  `key_requirements` — список ключевых требований для матчинга и `extractor_entities` — сохранённый `extractor_json` для backend search.
- `tests/fixtures/screening_scenarios.csv` — сценарии для проверки `screening_assistant`.
- `tests/fixtures/extractor_agent/` — кейсы для проверки `extractor_agent_runner.py`.
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
- Читает `tests/fixtures/screening_scenarios.csv`, поддерживает одиночные сценарии и chain-сценарии.
- Chain-группы зашиты в коде: `chain_salary_3x = [12,29,30]`, `chain_bot_check = [26,27]`, `chain_company_hidden = [23,24]`.
- Загружает вакансии из `tests/fixtures/cdm/*.json` и распределяет их по кейсам в режиме `round_robin_by_case`.
- Передает в prompt реальный контекст вакансии: `recruiter_name`, `candidate_name`, `title`, `company_name`, `responsibilities`, `work_format`, `location`, `firm_description`, `vacancy_url`, `salary`, `questions`.
- `work_format` в контекст прокидывается как raw backend value: `office`, `hybrid`, `remote`.
- Для сценариев `23/24` принудительно включается скрытый поиск: в контексте `Название компании: СКРЫТО`, `vacancy_url` очищается. Для открытого кейса `31` компания не скрывается.
- Генерирует реплики кандидата по сценарию, получает ответы `screening_assistant` и валидирует их через LLM-судью и набор deterministic-checks для критичных кейсов.
- CSV ожидает колонки: `Название сценария`, `Краткое описание сценария`, поле с ожидаемым поведением и поле с примерами диалогов.

Запуск:
```bash
python -m app.screening_scenarios_runner \
  --csv-path tests/fixtures/screening_scenarios.csv \
  --cdm-dir tests/fixtures/cdm \
  --messages-per-scenario 3
```

Точечный прогон:
```bash
python -m app.screening_scenarios_runner \
  --csv-path tests/fixtures/screening_scenarios.csv \
  --cdm-dir tests/fixtures/cdm \
  --scenario-indices 23,24,31 \
  --messages-per-scenario 3
```

Параметры:
- `--csv-path` — путь к CSV со сценариями.
- `--cdm-dir` — директория с CDM-фикстурами вакансий.
- `--scenario-indices` — список индексов строк CSV через запятую (например `23,24`) для точечного прогона.
- `--messages-per-scenario` — SINGLE: сообщений на сценарий; CHAIN: прогонов диалога на цепочку.
- `--max-scenarios` — лимит сценариев для быстрого прогона.

Отчеты:
- `tests/reports/screening_scenarios/screening_scenarios_report_<timestamp>.json`
- Структура отчета компактная: top-level `summary`, `cases`, `mismatches`.
- `cases` хранит подробности по single/chain-кейсам, `mismatches` — только неуспешные кейсы/раны для быстрого просмотра.
- Вместо полного dialog-context в отчет пишется сокращенный `vacancy_ref`: `title`, `company`, `vacancy_url`, `location`, `work_format`.

### `app/extractor_agent_runner.py` — тест пайплайна AI Search (`step1/2/3`)
Как работает:
- Step1: прогоняет запрос рекрутера через LLM-парсер и получает `extractor_json` строго по контракту (валидация ловит drift: лишние поля, неверные форматы, неверные значения).
- Step2: делает только маппинг `extractor_json -> payload` для backend `/site/searchBool` (без «добавления смысла» и без словарей/ID-маппингов).
- Step3: отправляет `payload` в backend `/site/searchBool` и читает `count`.
- Ответ backend `400 Positions or skills or keys must be set` классифицируется как `insufficient_search_terms` (не считается падением Step3).

Источники кейсов:
- `real`: кейсы из `tests/fixtures/extractor_agent` (обычно `amp_*`).
- `suite`: встроенные регрессионные кейсы (в коде), гарантируют наличие якоря (`positions/skills/keywords`).
- `syn`: синтетические деградированные запросы (генерируются из `real+suite` без добавления смысла).

Конфигурация:
- Prompt берется из `tests/tools/model.yaml` (поддерживается блок `extractor_agent.prompt_id/prompt_version`, а также fallback к `top-level/prompt.*`).
- Можно переопределить через `--prompt-id/--prompt-version` или env `EXTRACTOR_AGENT_PROMPT_ID/EXTRACTOR_AGENT_PROMPT_VERSION`.
- Для Step3 по умолчанию используются env: `AI_SEARCH_BASE_URL`, `AI_SEARCH_AUTH_TOKEN`.

Запуск (полный `1:1:1` прогон):
```bash
export OPENAI_API_KEY=sk-...
export AI_SEARCH_AUTH_TOKEN=...
export AI_SEARCH_BASE_URL=https://...

python app/extractor_agent_runner.py \
  --cases-dir tests/fixtures/extractor_agent \
  --cases-count 20 \
  --suite-count 20 \
  --synthetic-count 20 \
  --mix-ratios real=1,suite=1,syn=1 \
  --mix-seed 42 \
  --steps 1,2,3
```

Только Step1 (проверка промпта):
```bash
python app/extractor_agent_runner.py \
  --cases-dir tests/fixtures/extractor_agent \
  --cases-count 20 \
  --suite-count 20 \
  --synthetic-count 20 \
  --mix-ratios real=1,suite=1,syn=1 \
  --mix-seed 42 \
  --steps 1
```

Отчеты:
- `tests/reports/extractor_agent_full/extractor_agent_full_report_<run_id>.json`
- В каждом кейсе сохраняются `extractor_json` и `step3_payload` (распашенно) для дебага маппинга.

### `app/enrich_cdm_with_extractor_entities.py` — заполнение `vacancy.extractor_entities` в CDM
Как работает:
- Берёт `vacancy.title` из каждого `cdm_*.json`.
- Прогоняет title через prompt `extractor_agent`.
- Валидирует полученный `extractor_json` по контракту Step1.
- Записывает результат в `vacancy.extractor_entities`.

Зачем нужен:
- Чтобы не вызывать `extractor_agent` заново при каждом запуске `sourcing_assistant_runner.py`.
- Чтобы backend search для `sourcing_assistant_runner.py` строился из уже сохранённых сущностей, а не из сырого title.

Запуск:
```bash
# Все CDM
python app/enrich_cdm_with_extractor_entities.py

# Только первые 3 CDM
python app/enrich_cdm_with_extractor_entities.py --cdm-count 3

# С переопределением prompt
python app/enrich_cdm_with_extractor_entities.py \
  --prompt-id pmpt_... \
  --prompt-version 31
```

Важно:
- Скрипт перезаписывает только `vacancy.extractor_entities`.
- JSON пишется в UTF-8 без BOM.

### `app/sourcing_assistant_runner.py` — тест `sourcing_assistant` на реальных кандидатах из backend
Как работает:
- Берёт требования из `vacancy.key_requirements` или альтернативного источника (`stack_skills` / `responsibilities_parser`).
- Берёт backend search-структуру из `vacancy.extractor_entities`.
- Через `build_step3_payload(...)` собирает payload для backend `/site/searchBool`.
- Получает кандидатов, выбирает нужное количество, прогоняет каждого через `sourcing_assistant` и сравнивает `passed` с детерминированным baseline.

Важно:
- Если в CDM нет `vacancy.extractor_entities`, раннер не сможет построить корректный backend search payload.
- Если после изменения prompt `extractor_agent` нужно обновить сущности в CDM, сначала заново запустите `app/enrich_cdm_with_extractor_entities.py`.

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
- По умолчанию также прогоняет deterministic regression-кейсы для `work_format`, которые строятся из реальных вакансий `tests/fixtures/cdm/*`.
- Прогоняет диалоги через `screening_autofill` и парсит JSON.
- Валидирует схему, семантику и точечные ожидаемые поля для regression-кейсов.
- `prompt_id` берется из CLI/env/model.yaml, без него запуск невозможен.

Запуск:
```bash
python -m app.screening_autofill_runner --cdm-count 5 --variants-per-cdm 3

# Только regression-кейсы по реальным CDM
python -m app.screening_autofill_runner --regression-only --regression-variants-per-case 3

# Обычный запуск без regression-кейсов и без flatten-like-prod
python -m app.screening_autofill_runner --no-include-regression-cases --no-flatten-like-prod

# Несколько выбранных regression-сценариев, по 2 варианта на каждый
python -m app.screening_autofill_runner \
  --regression-only \
  --regression-case-names wf_hybrid_explicit_candidate,wf_empty_when_only_recruiter_mentions_hybrid \
  --regression-variants-per-case 2
```

Параметры:
- `--cdm-dir` — путь к CDM.
- `--cdm-count` — сколько CDM взять в исходный пул вакансий (по умолчанию все).
- `--variants-per-cdm` — число диалогов на CDM.
- `--noise-level` — 0..2, уровень шума в ответах.
- `--allow-two-questions` — разрешить два вопроса в реплике рекрутера.
- `--flatten-like-prod` / `--no-flatten-like-prod` — преобразовать диалог в одну строку (по умолчанию включено).
- `--seed` — сид для вариативности.
- `--dialogue-gen-model` — модель генерации диалогов.
- `--prompt-id`, `--prompt-version` — переопределить промпт.
- `--include-regression-cases` / `--no-include-regression-cases` — добавлять встроенные regression-кейсы к обычному CDM-прогону (по умолчанию включено).
- `--regression-only` — запускать только regression-кейсы, без обычной генерации диалогов.
- `--regression-case-names` — список имён regression-кейсов через запятую для точечного прогона.
- `--regression-variants-per-case` — сколько вариантов диалога строить на каждый regression-кейс.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/screening_autofill/screening_autofill_report_<run_id>.json`
- Отчёт теперь компактный: `summary`, краткий `cases` и подробные `mismatches`.
- Для regression-кейсов в `cases/mismatches` видны `expected_work_format` / `actual_work_format` и `source_cdm`.

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

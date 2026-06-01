# AI Agents Workspace

Набор CLI-раннеров для тестирования рекрутмент-промптов и связанных prompt-пайплайнов. Основной текущий способ прогона - специализированные раннеры по отдельным компонентам: `screening_scenarios_runner`, `screening_guardrails_runner`, `extractor_agent_runner`, `first_touch_runner`, `first_touch_event_runner`, `message_classifier_runner`, `screening_autofill_runner`, `verdict_classifier_runner`, `responsibilities_parser_runner`, `one_line_search_query_builder_runner`, `sourcing_assistant_runner`.

## Структура проекта
- `app/` — CLI-раннеры.
- `adapters/` — преобразования CDM -> input-форматы промптов.
- `cdm/` — схема CDM и каталог `samples/` (может быть пустым).
- `messageLabelGenerator/` — обвязка промпта `message_classifier`.
- `screeningAssistant/` — обвязка промпта `screening_assistant`.
- `screening_autofill/` — обвязка промпта `screening_autofill`.
- `verdict_classifier/` — обвязка промпта `verdict_classifier`.
- `tests/fixtures/` — фикстуры (CDM, screening-сценарии, extractor-кейсы).
- `tests/tools/` — `model.yaml` и утилиты для генерации baseline-фикстур.
- `tests/reports/` — отчёты раннеров в отдельных подкаталогах по имени раннера (`screening_scenarios`, `screening_guardrails`, `message_classifier`, `first_touch`, `first_touch_event`, `screening_autofill`, `verdict_classifier`, `extractor_agent_full`, `responsibilities_parser`, `one_line_search_query_builder`, `sourcing_assistant`, `runs` и др.).

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

Создайте `.env` (можно скопировать из `.env.example`) и задайте как минимум:
```bash
OPENAI_API_KEY=sk-...

# Для backend search раннеров:
AI_SEARCH_BASE_URL=https://...
AI_SEARCH_AUTH_TOKEN=...
```

Важно:
- Большинство раннеров не читают `.env` автоматически. Для них переменные должны быть выставлены в окружении текущего процесса/терминала.
- `.env` автоматически подхватывают только `app/first_touch_event_runner.py` и `app/enrich_cdm_with_extractor_entities.py`.
- `OPENAI_API_KEY` нужен всем раннерам, которые обращаются к LLM.
- `AI_SEARCH_BASE_URL` и `AI_SEARCH_AUTH_TOKEN` нужны для backend-шагов в `extractor_agent_runner.py`, `one_line_search_query_builder_runner.py`, `sourcing_assistant_runner.py`.
- `OPENAI_BASE_URL` опционален и используется только `app/extractor_agent_runner.py` для Step1.

## Конфигурация промптов и профилей
`tests/tools/model.yaml` хранит `prompt_id`/`prompt_version` для:
- `first_touch`
- `first_touch_event_invite`
- `message_classifier`
- `screening_assistant`
- `screening_autofill`
- `verdict_classifier`
- `extractor_agent`
- `one_line_search_query_builder`
- `responsibilities_parser`
- `sourcing_assistant`
- `candidate_simulator` (профили кандидатов; использовались только удалённым `app/runner.py` — сейчас не задействованы ни одним раннером)

Переменные окружения для переопределения:
- `FIRST_TOUCH_PROMPT_ID` — для `app/first_touch_runner.py`.
- `FIRST_TOUCH_EVENT_PROMPT_ID`, `FIRST_TOUCH_EVENT_PROMPT_VERSION` — для `app/first_touch_event_runner.py`.
- `MESSAGE_CLASSIFIER_PROMPT_ID`, `MESSAGE_CLASSIFIER_PROMPT_VERSION` — для `app/message_classifier_runner.py`.
- `SCREENING_AUTOFILL_PROMPT_ID`, `SCREENING_AUTOFILL_PROMPT_VERSION` — для `app/screening_autofill_runner.py`.
- `VERDICT_CLASSIFIER_PROMPT_ID`, `VERDICT_CLASSIFIER_PROMPT_VERSION` — для `app/verdict_classifier_runner.py`.
- `EXTRACTOR_AGENT_PROMPT_ID`, `EXTRACTOR_AGENT_PROMPT_VERSION` — для `app/extractor_agent_runner.py`.
- `ONE_LINE_SEARCH_QUERY_BUILDER_PROMPT_ID`, `ONE_LINE_SEARCH_QUERY_BUILDER_PROMPT_VERSION` — для builder prompt в `app/one_line_search_query_builder_runner.py`.
- `RESPONSIBILITIES_PARSER_PROMPT_ID`, `RESPONSIBILITIES_PARSER_PROMPT_VERSION` — для `app/responsibilities_parser_runner.py` и для режима `--requirements-source responsibilities_parser` в `app/sourcing_assistant_runner.py`.
- `SOURCING_ASSISTANT_PROMPT_ID`, `SOURCING_ASSISTANT_PROMPT_VERSION` — для `app/sourcing_assistant_runner.py`.

Важно:
- `app/screening_scenarios_runner.py` и `app/screening_guardrails_runner.py` читают `screening_assistant.prompt_id/prompt_version` только из `tests/tools/model.yaml`.

## Фикстуры и данные
- `tests/fixtures/cdm/` — CDM-фикстуры вакансий.
  Checked-in фикстуры в репозитории уже могут содержать дополнительные поля `vacancy.raw_vacancy`, `vacancy.key_requirements`, `vacancy.extractor_entities`.
  Baseline-CDM генерируются напрямую через `tests/tools/make_vacancies.py` (ранее это делала команда `python -m app.runner gen-fixtures`, которая удалена вместе с `app/runner.py`).
  Эти baseline-CDM подходят для smoke/e2e-сценариев, но не добавляют `raw_vacancy`, `key_requirements`, `extractor_entities`.
- `tests/fixtures/screening_scenarios.csv` — сценарии для проверки `screening_assistant`.
- `tests/fixtures/extractor_agent/` — кейсы для проверки `extractor_agent_runner.py` (сейчас хранятся в `cases.yaml`).
- `cdm/schema.json` — схема CDM.

Дополнительные требования к данным:
- `responsibilities_parser_runner.py` и `one_line_search_query_builder_runner.py` требуют `vacancy.raw_vacancy`.
- `sourcing_assistant_runner.py` использует `vacancy.extractor_entities` для backend search и предпочитает `vacancy.key_requirements` как основной источник требований.
- `app/enrich_cdm_with_extractor_entities.py` заполняет только `vacancy.extractor_entities`; поля `raw_vacancy` и `key_requirements` он не генерирует.

## Раннеры (`app/`)

> Примечание: исторический end-to-end раннер `app/runner.py` удалён как неиспользуемый.
> Генерация baseline-CDM (бывшая команда `python -m app.runner gen-fixtures`) теперь
> выполняется напрямую через `tests/tools/make_vacancies.py`.

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
- Для Step1 опционально поддерживается `OPENAI_BASE_URL` (по умолчанию `https://api.openai.com/v1`).

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
- Если выбран `--requirements-source cdm_key_requirements`, но `vacancy.key_requirements` отсутствует, раннер fallback-ится на `stack_skills`.
- Если выбран `--requirements-source responsibilities_parser` и LLM-парсер не смог извлечь требования, раннер также fallback-ится на `cdm_key_requirements`/`stack_skills`.

Запуск:
```bash
export OPENAI_API_KEY=sk-...
export AI_SEARCH_AUTH_TOKEN=...
export AI_SEARCH_BASE_URL=https://...

python -m app.sourcing_assistant_runner \
  --cdm-count 5 \
  --candidate-sample-size 10
```

Запуск с альтернативным источником требований:
```bash
python -m app.sourcing_assistant_runner \
  --cdm-count 5 \
  --requirements-source responsibilities_parser \
  --report-verbosity standard \
  --sample-mode random
```

Параметры:
- `--cdm-dir`, `--cdm-count` — путь к CDM и лимит выбранных вакансий.
- `--cases-count` — случайно выбрать N вакансий из уже отобранного пула.
- `--seed` — сид выборки и прогона.
- `--prompt-id`, `--prompt-version` — переопределить prompt `sourcing_assistant`.
- `--requirements-source` — `cdm_key_requirements`, `stack_skills` или `responsibilities_parser`.
- `--report-verbosity` — `compact`, `standard`, `full`.
- `--base-url`, `--step3-path`, `--token`, `--timeout-s`, `--step3-retries`, `--token-in-body`, `--token-in-header` — настройки backend `/site/searchBool`.
- `--candidate-pool-size` — сколько backend-профилей запрашивать на вакансию.
- `--candidate-sample-size` — сколько из найденных профилей прогонять через `sourcing_assistant`.
- `--sample-mode` — `first` или `random`.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/sourcing_assistant/sourcing_assistant_report_<run_id>.json`

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
- Классифицирует и считает accuracy/матрицу ошибок.
- Генерирует сообщения на основе CDM-контекста и сценарных подсказок, но с заранее известным target-классом.

Запуск:
```bash
python -m app.message_classifier_runner --messages-per-class 3 --seed 42
```

Параметры:
- `--cdm-dir`, `--cdm-count` — путь к CDM и лимит вакансий для контекста генерации.
- `--messages-per-class` — сколько сообщений сгенерировать на каждый класс (обязательный параметр).
- `--noise-level` — уровень косвенности/шума `0..2`.
- `--seed` — сид для генерации/перемешивания.
- `--message-gen-model` — модель для генерации synthetic candidate messages.
- `--prompt-id`, `--prompt-version` — переопределить prompt `message_classifier`.
- `--scenario-mode` — `random` или `cycle` для выбора сценарных подсказок.
- `--scenario-count-per-class` — ограничить число сценарных подсказок на класс.
- `--max-attempts-multiplier` — лимит попыток генерации.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/message_classifier/message_classifier_report_<run_id>.json`

### `app/first_touch_runner.py` — тест первого касания
Как работает:
- Строит `InputForm` из CDM и генерирует сообщение через сохранённый prompt `first_touch`
  (встроенный `FirstTouchGenerator`, по аналогии с `first_touch_hh_runner`; контракт
  input-переменных и постобработка подписи сохранены от прежнего внешнего генератора).
- Проверяет наличие фактов и галлюцинаций с помощью LLM-оценщика.
- Опционально требует вопрос в сообщении.
- `prompt_id` берётся из `FIRST_TOUCH_PROMPT_ID`/`model.yaml` (секция `first_touch`).

Запуск:
```bash
python -m app.first_touch_runner --limit 5 --require-question
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
- `tests/reports/first_touch/first_touch_report_<run_id>.json`

### `app/first_touch_event_runner.py` — простой раннер для event-invite prompt
Как работает:
- Напрямую вызывает сохранённый prompt по `prompt_id/prompt_version`, без CDM и без построения `InputForm`.
- На вход подаёт только `candidate_name` в JSON (`{"candidate_name": ...}`).
- Генерирует сообщения для набора имён, отдельно прогоняет кейс с пустым именем.
- Проверяет каждое сообщение на обязательные факты, выдуманные детали и финальный вопрос про ссылку на регистрацию.
- Отдельно считает вариативность по телу сообщения без приветствия и по последнему вопросу.

Запуск:
```bash
python -m app.first_touch_event_runner

# Точечный прогон на своём наборе имён
python -m app.first_touch_event_runner \
  --names "Анна,Мария,Илья,Олег" \
  --repeats-per-name 2
```

Параметры:
- `--prompt-id`, `--prompt-version` — переопределить prompt.
- `--eval-model` — модель-судья (по умолчанию `gpt-4.1-mini`).
- `--names` — список имён через запятую.
- `--repeats-per-name` — сколько генераций делать на одно имя.
- `--include-empty-name` / `--no-include-empty-name` — включать ли кейс с пустым `candidate_name`.
- `--min-unique-bodies`, `--min-unique-questions` — пороги для `variability_passed`.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/first_touch_event/first_touch_event_report_<run_id>.json`

### `app/responsibilities_parser_runner.py` — тест `responsibilities_parser`
Как работает:
- Берёт `vacancy.raw_vacancy` из CDM-фикстур.
- Вызывает prompt `responsibilities_parser` и ожидает строгий JSON-массив строк.
- Сверяет извлечённые пункты с текстом вакансии, `vacancy_stack` и `vacancy_skills`.
- Проверяет prompt-contract: формат ответа, дубли, число пунктов, покрытие терминов и наличие совпадений с ожидаемыми требованиями.

Запуск:
```bash
python -m app.responsibilities_parser_runner --cdm-count 10

# Более подробный отчёт на выборке из 5 вакансий
python -m app.responsibilities_parser_runner \
  --cases-count 5 \
  --report-verbosity standard \
  --min-total-matches 3
```

Параметры:
- `--cdm-dir`, `--cdm-count` — путь к CDM и лимит выбранных вакансий.
- `--cases-count` — случайно выбрать N вакансий из доступного пула.
- `--seed` — сид выборки.
- `--prompt-id`, `--prompt-version` — переопределить prompt `responsibilities_parser`.
- `--min-total-matches` — минимальный порог совпадений с `vacancy_stack U vacancy_skills`.
- `--no-require-all-in-text` — отключить строгую проверку, что каждый предсказанный пункт встречается в тексте вакансии.
- `--report-verbosity` — `compact`, `standard`, `full`.
- `--quiet` — без прогресс-логов.

Отчеты:
- `tests/reports/responsibilities_parser/responsibilities_parser_report_<run_id>.json`

### `app/one_line_search_query_builder_runner.py` — тест one-line search query builder
Как работает:
- Step1: строит однострочный поисковый запрос по `raw_vacancy`.
- Step2: прогоняет этот запрос через `extractor_agent` и валидирует промежуточный `extractor_json`.
- Step3: собирает payload для `/site/searchBool`, отправляет его в backend и сохраняет результаты семантической проверки.
- Поддерживает частичный прогон по шагам: `1`, `1,2`, `1,2,3`.

Запуск:
```bash
export OPENAI_API_KEY=sk-...
export AI_SEARCH_AUTH_TOKEN=...
export AI_SEARCH_BASE_URL=https://...

python app/one_line_search_query_builder_runner.py \
  --cdm-count 10 \
  --steps 1,2,3
```

Только Step1 и Step2:
```bash
python app/one_line_search_query_builder_runner.py \
  --cdm-count 10 \
  --steps 1,2 \
  --report-mode full
```

Параметры:
- `--cdm-dir`, `--cdm-count` — путь к CDM и лимит вакансий.
- `--steps` — один из режимов: `1`, `1,2`, `1,2,3`.
- `--cfg` — путь к `tests/tools/model.yaml`.
- `--builder-prompt-id`, `--builder-prompt-version` — переопределить prompt `one_line_search_query_builder`.
- `--extractor-prompt-id`, `--extractor-prompt-version` — переопределить prompt `extractor_agent` для Step2.
- `--model` — модель для prompt-вызовов.
- `--base-url`, `--step3-path`, `--token`, `--timeout-s`, `--step3-retries`, `--token-in-body`, `--token-in-header` — настройки backend `/site/searchBool`.
- `--only-russian`, `--only-english`, `--only-with-contacts`, `--only-with-higher-education`, `--current-position-title` — backend-флаги фильтрации.
- `--limit`, `--offset`, `--shuffle`, `--highlight` — дополнительные backend-параметры выборки.
- `--report-dir`, `--report-mode`, `--report-json-indent` — управление путём и детализацией отчёта.

Отчеты:
- `tests/reports/one_line_search_query_builder/one_line_search_query_builder_report_<run_id>.json`

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

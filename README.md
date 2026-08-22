# AI Agents — QA-харнесс рекрутинговых промптов

Тестовый стенд (**НЕ продукт**) для регрессионной проверки промптов рекрутингового AI-ассистента.
Промпт-под-тестом берётся из репозитория [`prompts`](docs/LOCAL_PROMPTS.md) (приоритетный путь) или из
stored-промптов OpenAI; раннеры гоняют его по сценариям и оценивают ответы. Всё на русском, Python 3.12.

«Тесты промптов» = **CLI-раннеры** (единственный вид тестов здесь). Юнит-тестов (`pytest`) на код
харнесса нет: корректность раннеров проверяется их `--offline`-режимами и прогоном глазами, а
структурный контракт `qa_harness ⊥ app` — линтером импортов.

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate     # WSL/Linux (рабочее окружение)
#   Windows: python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e .[dev]                                  # qa_harness + dev-инструменты (import-linter)

export OPENAI_API_KEY=sk-...                           # .env раннеры НЕ читают, см. «Грабли»
python -m qa_harness.runners.message_classifier --offline    # прогон без сети — проверка, что всё встало
lint-imports                                                # гейт изоляции: qa_harness не зависит от app/
```

## Архитектура

Целевое дерево — устанавливаемый пакет **`qa_harness`** (src-layout). Миграция раннеров **завершена**:
все раннеры живут в `qa_harness`. Легаси-дерево `app/` ещё рабочее и остаётся до cutover.

| Путь | Что там |
|---|---|
| `src/qa_harness/core/` | инфраструктура: `llm_client`, `prompt_source`, `config`, `usage`, `jsonio`, `metrics`, `reporting`, `run_loop`, `cdm` |
| `src/qa_harness/domain/` | рекрутинговая логика: `judge`, `generators`, `classifiers`, `extractor`, `screening*`, `sourcing`, `text` |
| `src/qa_harness/pipeline/` | конвейер AI-поиска: step1 parse → step2 payload → step3 backend |
| `src/qa_harness/runners/` | тонкие CLI-раннеры (`python -m qa_harness.runners.<name>`) |
| `app/`, `adapters/`, `screeningAssistant/` | **ЛЕГАСИ** до cutover — [docs/legacy_runners.md](docs/legacy_runners.md). Не удалять, новый код не добавлять |
| `tests/fixtures/` | данные: golden-кейсы, сценарии, рецепты, констрейнты генерации |
| `tests/tools/model.yaml` | **источник правды** по промптам (`prompt_id`/`prompt_version`/`local_component`) |
| `tests/reports_v2/` | отчёты новых раннеров (`tests/reports/` — легаси) |

## Документация

| Документ | Жанр | О чём |
|---|---|---|
| [docs/LOCAL_PROMPTS.md](docs/LOCAL_PROMPTS.md) | how-to | Переключатель источника промптов `stored ↔ local`, установка пакета `prompts`, Docker |
| [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md) | reference | Контракт отчёта (три файла), поля `meta`/`summary`/`metrics`/`cases` |
| [docs/screening_split/](docs/screening_split/) | playbook + бэклог | Раздельный скрининг: как читать отчёт, открытые пункты, журналы прогонов |
| [docs/legacy_runners.md](docs/legacy_runners.md) | reference | Легаси `app/`-раннеры (заморожено до cutover) |

Полный индекс — [docs/README.md](docs/README.md).

## Окружение

Переменные (шаблон — `.env.example`):

| Переменная | Кому нужна |
|---|---|
| `OPENAI_API_KEY` | всем LLM-раннерам |
| `OPENAI_BASE_URL` | опционально — кастомный OpenAI-совместимый эндпоинт |
| `AI_SEARCH_BASE_URL` | backend-поиск (`{URL}/site/searchBool`); тест-стенд — `https://testsecond.hlebusheck.ru` |
| `AI_SEARCH_AUTH_TOKEN` | токен backend; передаётся **в теле** (флаг `--token-in-body`) |
| `QA_HARNESS_PROMPT_SOURCE` | `stored`\|`local` — то же, что флаг `--prompt-source` |
| `PROMPTS_REPO_PATH` | ДЕВ-путь к исходникам репо `prompts` (то же, что `--prompts-path`) |

⚠️ `https://testsecond.podbor.io/search` — это веб-UI (редирект на `/auth`), **не** API.

## Источник промпта: `stored` ↔ `local`

У всех LLM-раннеров есть общие флаги (`core.add_prompt_source_args`):
`--prompt-source {stored,local}` · `--local-prompt-version vN` · `--prompts-path PATH`.

- **`local`** — ⭐ приоритетный путь: тело из `system.md` и параметры из `config.yaml` пакета `prompts`
  (ставится как релиз — wheel из GitHub Release/GHCR). Единый источник правды прод↔тесты.
  OpenAI выключает `v1/prompts` (де-приоритизация 03.06.2026, отключение 30.11.2026).
- **`stored`** — дефолт для обратной совместимости: `responses.create(prompt={id, version})`.

`prompt_id`/`prompt_version` — **только из `tests/tools/model.yaml`**, не хардкодить в раннерах.
Точечный override — env `<COMPONENT>_PROMPT_ID/_VERSION` или флаги `--prompt-id/--prompt-version`
(там, где раннер их поддерживает). Детали и грабли — [docs/LOCAL_PROMPTS.md](docs/LOCAL_PROMPTS.md).

## Режимы входа

Источник кейсов выбирается флагом; набор флагов у раннеров РАЗНЫЙ (точно — `--help`):

- **golden** — курируемые данные, детерминизм, годится в CI-гейт. Это **дефолт** почти везде; явный
  флаг `--golden` есть там, где раннер умеет и генерацию (`first_touch`, `first_touch_event`,
  `one_line_search_query_builder`, `responsibilities_parser`, `screening_autofill`,
  `screening_guardrails`, `sourcing_assistant`);
- **`--generate`** — вариативная LLM-генерация входов, проверка робастности промпта на РАЗНЫХ входах.
  При `temperature > 0` **недетерминирован → в гейт не годится.** Есть у `first_touch`,
  `one_line_search_query_builder`, `responsibilities_parser`, `screening_autofill`,
  `screening_guardrails`, `screening_scenarios`, `screening_split`, `sourcing_assistant`;
  у классификаторов роль генерации играет `--mode synthetic|all`;
- **`--offline`** — replay/эвристика без сети (CI-дымовой прогон, демо). Есть у всех, кроме
  `extractor_agent` (golden-only) и `screening_counters`.

Движок генерации — `domain/generators` (цикл produce → validate → retry → fallback + трасса).
Три типа producer: адаптивный LLM-кандидат (`CandidateAgent`), seeded-сообщение/диалог с известной
меткой, генератор контекста (`VacancyGenerator`). **`fallback` — опция per-runner:** включён там, где
запасной вариант безвреден, и ВЫКЛючен для размеченных данных (классификаторы/autofill/
responsibilities) — там лучше недодать (`errors`), чем влить мислейбл в метрику. Исчерпание попыток
даёт `source=failed` (не исключение) → кейс уходит в `errors`.

Три модели разведены: генератор (`gpt-4.1-mini`) ≠ судья ≠ промпт-под-тестом. Констрейнты, персоны и
словари — данные в `tests/fixtures/generation/<runner>/`.

## Отчёты

Каждый прогон пишет **три файла** в `tests/reports_v2/<runner>/` (UTF-8 без BOM, `ensure_ascii=False`):

- `*.metrics.json` — `meta` + `summary` + `metrics` + `failures_index` (для дашбордов, трендов, CI);
- `*.cases.json` — по кейсам: входы, транскрипт/артефакты, вердикт (для разбора падений);
- `*.review.md` — человекочитаемый рендер из тех двух (у `screening_split` отключён).

**`summary.passed/failed` = качество промпта. Инфра-сбои (timeout/auth/http/SSL) идут в
`summary.errors`, а не в `failed`** — флакот бэкенда не портит сигнал качества.
Полный контракт — [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md).

## Раннеры

Запуск: `python -m qa_harness.runners.<name>`. Точный список флагов — `--help` и докстринг модуля
(источник правды); ниже — назначение и канонический вызов.

### Классификаторы

| Раннер | Что проверяет | Оценка |
|---|---|---|
| `message_classifier` | реплика кандидата → класс (`reason_farewell`/`no_reason`/`acceptance`/`human_needed`) | `LabelJudge` (точная метка), accuracy + per-class + confusion + `by_split` |
| `verdict_classifier` | полный диалог → вердикт (`passed`/`failed`/`deadlock`) | то же |

```bash
python -m qa_harness.runners.message_classifier --offline                 # без сети
python -m qa_harness.runners.verdict_classifier --mode all --dialogs-per-verdict 3 --seed 42
```
Режимы: `--mode regression|synthetic|all`.

### Конвейер AI-поиска

| Раннер | Что проверяет | Оценка |
|---|---|---|
| `extractor_agent` | фраза рекрутера → сущности → payload → backend `count` | поэтапная, **без LLM-судьи** |
| `one_line_search_query_builder` | вакансия → однострочный boolean-запрос → extractor → backend | поэтапная: format + no_leakage + golden-семантика |
| `sourcing_assistant` | консервативная 0/1-оценка резюме по требованиям | контракт + семантика по засеянным `expect_passed` |

`extractor_agent` оценивает каждый шаг отдельно: **step1** = contract (форма JSON) + semantic (golden:
попали ли термины в нужные bucket'ы, не уехал ли город в `positions`) + format (голый ли JSON);
**step2** = mapping integrity (ничего не потеряно при сборке payload); **step3** = retrieval
(`count` — это **информация**, не pass/fail). Итог кейса `passed = contract & semantic & mapping`;
сбои бэкенда → `errors`. Кейсы — курируемые якоря `tests/fixtures/extractor_agent/anchors.yaml`
(фраза + `expect`/`forbid`, эталон проставляет человек).

```bash
python -m qa_harness.runners.extractor_agent --steps 1                      # тест ПРОМПТА без backend
python -m qa_harness.runners.extractor_agent --steps 1,2,3 --token-in-body  # + count из backend
python -m qa_harness.runners.sourcing_assistant --count-only                # быстрый триаж (limit=0)
```
Долгие прогоны: `--workers`, раздельные `--step1-timeout`/`--step3-timeout`, `--backend-fail-fast`,
`--checkpoint-every`, сохранение частичного отчёта по Ctrl+C (оркестрация — `core.run_loop`).

### Первое касание и парсеры

| Раннер | Что проверяет |
|---|---|
| `first_touch` (+ `first_touch_hh`, `first_touch_event`) | генерация первого касания: факты vs галлюцинации (LLM-судья) + эвристики (приветствие, абзацы, запрет маркеров) |
| `responsibilities_parser` | текст вакансии → ключевые термины: контракт + golden-семантика |
| `screening_autofill` | диалог → форма скрининга: контракт + golden-ожидания + анти-утечка |

### Скрининг — монолит

| Раннер | Что проверяет |
|---|---|
| `screening_scenarios` | сценарии из CSV → живой `screening_assistant` → `ScenarioJudge` против `expected_behavior` |
| `screening_guardrails` | мультитёрн-разговор → ловит `self_answer`, `repeated_questions`, `premature_end` |

```bash
python -m qa_harness.runners.screening_scenarios --component screening_assistant_hh --sample 5
```
Онлайн гоняются только сценарии с примерами диалога кандидата (base CSV: 10/64, hh CSV: 0/50).
Chain-групп здесь нет — они остались зашитыми в легаси `app/screening_scenarios_runner.py`.

### Скрининг — раздельный (split)

Вместо монолита: **Аналитик** (`screening_analyzer`, строгий JSON `Decision`) + **Интервьюер**
(`screening_interviewer`, одно сообщение) + КОД-оркестратор, портированный 1:1 из tgApi в
`domain/screening_split/`. **LOCAL-only** — stored-эквивалента нет.

| Раннер | Что проверяет |
|---|---|
| `screening_split` | три слоя с атрибуцией ошибки: **A** детерминированные инварианты `Decision`/state (`scenario_checks.yaml`) · **B** `leak_scan`/`injection_scan` + `InterviewerJudge` (верно ли передан СМЫСЛ инструкции) · **C** `ScenarioJudge` — страховка для сценариев без инварианта |
| `screening_counters` | анти-луп: настойчивый переспрашиватель обязан завершиться в пределах кап, кооперативный → `FINISH` без ложного капа. Реальный Аналитик + фейковые Интервьюер/стор |

```bash
python -m qa_harness.runners.screening_split --prompts-path ../prompts --input-mode scripted
python -m qa_harness.runners.screening_split --prompts-path ../prompts --channel hh
python -m qa_harness.runners.screening_counters --prompts-path ../prompts
```

Режимы входа: `scripted` — реплики из `candidate_inputs.yaml` (детерминированный CI-гейт);
`generated` — адаптивный LLM-кандидат, засеянный из рецепта. Инвариант Аналитика гейтит в обоих.
Счётчики завершения — **в коде движка, не в LLM**: событийные пороги `_EVENT_STOP`, reask-cap
и универсальный `no_progress`-cap.

**Как читать отчёт — [docs/screening_split/report_analysis.md](docs/screening_split/report_analysis.md)**
(что значит `passed`, атрибуция роли, реальный баг промпта vs дрейф генератора).

## Фикстуры и данные

- `tests/fixtures/cdm/{std,hh}/` — CDM-вакансии; baseline генерит `tests/tools/make_vacancies.py`
  (без `raw_vacancy`/`key_requirements`/`extractor_entities` — их добавляют отдельно);
- `tests/fixtures/extractor_agent/anchors.yaml` — курируемые якоря + golden;
- `tests/fixtures/{message,verdict}_classifier/regression_cases.json` — размеченные regression-кейсы;
- `tests/fixtures/screening_scenarios.csv` (+ `_hh`) — сценарии монолитного скрининга;
- `tests/fixtures/screening_split{,_hh}/` — `scenarios.csv` + рецепты `candidate_inputs.yaml` +
  инварианты `scenario_checks.yaml` + `counter_loops.yaml`;
- `tests/fixtures/generation/<runner>/` — констрейнты и персоны для `--generate`;
- `cdm/schema.json` — схема CDM.

## Грабли

- **Раннеры НЕ читают `.env`** — экспортируй в окружение: `set -a; source .env; set +a`.
- **`store=False` во всех вызовах харнесса** (`core.llm_client.STORE_RESPONSES`): в Responses API
  `store` по умолчанию `true`, из-за чего прогоны засоряли `platform.openai.com/logs`, где смотрят
  ПРОД. **Исключение — мультитёрн через `conversation=`** (`domain/screening`,
  `domain/screening_split/interviewer.py`): при `store=false` новые input/output не дописываются в
  conversation → история ходов теряется, поэтому там store остаётся продовым.
- **`extractor` step3: по умолчанию `limit=0` (только count).** Backend отдаёт `count` за ~11 с, а
  тяжёлые `profiles` (`limit>0`) — минутами/таймаут. Не ставь большой `--step3-limit` без нужды.
- **Профили `sourcing_assistant` таймаутят на ШИРОКИХ вакансиях** (высокий `count`) — это не настройка.
  Триаж: `--count-only` (~секунды) → полный прогон по узким. `SSLEOFError`/`Max retries` —
  транзиентный обрыв соединения (не таймаут), лечится повтором.
- **Ответ `/site/searchBool` — не только число:** `{count, profiles: [...]}`. `limit=0` → только
  `count`; `limit=N>0` → до N реальных профилей в `profiles`. `call_backend_search_bool(...)`
  возвращает `(kind, status, attempts, count, error, json)`, где 6-й элемент — полный ответ.
- **Окружение машины:** рабочий venv — Linux/WSL. Если терминал MSYS/git-bash, гоняй через
  `wsl.exe -e bash -lc '... source .venv/bin/activate ...'`.
- **Легаси sourcing-раннеры** требуют в CDM `vacancy.extractor_entities`/`raw_vacancy`
  (`python app/enrich_cdm_with_extractor_entities.py`); `make_vacancies.py` их не генерит.

## Конвенции

- Всё (промпты, сценарии, golden, комментарии, коммиты) — на русском. Сохраняй язык.
- JSON-отчёты и фикстуры — UTF-8 **без BOM**, `ensure_ascii=False`.
- Меняя версию промпта — правь `tests/tools/model.yaml`, не хардкодь.
- Новый код — в `qa_harness`, не в `app/`; `qa_harness` не должен импортировать `app/`/`adapters/`
  (проверяет `lint-imports`).

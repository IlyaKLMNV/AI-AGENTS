# AI Agents — QA-харнесс рекрутинговых промптов

Тестовый стенд (НЕ продукт) для регрессионной проверки промптов рекрутингового AI-ассистента.
Продуктовые промпты хранятся в OpenAI как stored prompts (`prompt_id` + `prompt_version`; реестр —
`tests/tools/model.yaml`). Всё на русском. Python 3.12. «Тесты промптов» = CLI-раннеры (единственный вид
тестов здесь): корректность раннеров проверяется их `--offline`-режимами и прогоном глазами. Юнит-тестов
(`pytest`) на код харнесса НЕ держим — структурный контракт `qa_harness ⊥ app` проверяет `lint-imports`.

## Архитектура (идёт миграция — два дерева)
- **`src/qa_harness/`** — НОВАЯ архитектура (целевая), устанавливается `pip install -e .`:
  - `core/` — инфраструктура (llm_client, config, usage, jsonio, metrics, reporting, cdm);
  - `domain/` — рекрутинг (judge, generators, classifiers, extractor, text);
  - `pipeline/` — AI-поиск (step1 parse → step2 payload → step3 backend);
  - `runners/` — тонкие раннеры (`python -m qa_harness.runners.<name>`).
- **`app/`** — ЛЕГАСИ-раннеры (ещё рабочие, `python -m app.<runner>`). **НЕ удалять до cutover.**
- `adapters/`, `screeningAssistant/screeningAss.py` — нужны легаси `screening_*`-раннерам.
- `docs/` — `REFACTOR_PLAN.md`, `REPORT_SCHEMA.md`, `MIGRATION_STATUS.md` (статус по раннерам),
  `EXTRACTOR_REDESIGN.md`. `tests/tools/model.yaml` — источник правды по промптам.

Переведены на новую архитектуру: **message_classifier, verdict_classifier, extractor_agent, one_line_search_query_builder, sourcing_assistant, responsibilities_parser, screening_autofill, first_touch (base), first_touch_hh, first_touch_event, screening_guardrails, screening_scenarios (+`--component screening_assistant_hh`)**.
Миграция раннеров завершена — все на `qa_harness`; легаси `app/` остаётся до cutover.

## Новые раннеры
Запуск `python -m qa_harness.runners.<name>`; отчёт — три файла в `tests/reports_v2/<runner>/`
(`*.metrics.json` + `*.cases.json` + человекочитаемый `*.review.md`, схема — `docs/REPORT_SCHEMA.md`). `summary.passed/failed` =
**качество промпта**; инфра-сбои → `summary.errors` (не в `failed`).
- классификаторы (message/verdict): `LabelJudge` (метка), accuracy + confusion + by_split; `--offline` без сети.
- extractor: контракт + **семантика по golden** (`anchors.yaml`), без LLM-судьи; поэтапные вердикты
  (step1/step2/step3); конкурентность + fail-fast + чекпоинты. См. `docs/EXTRACTOR_REDESIGN.md`.

## Источник промптов: platform.openai.com ↔ репозиторий `prompts` (переключатель)
Промпт-под-тестом берётся из одного из двух источников, переключение — одним ключом (см.
`docs/LOCAL_PROMPTS.md`, ядро — `src/qa_harness/core/prompt_source.py`):
- **local** — ⭐ приоритетный способ. Пакет `prompts` (репозиторий podbor/prompts), ставится как релиз
  (wheel из GHCR-образа/GitHub Release): тело из `system.md` + параметры из `config.yaml`, вызов
  `responses.create(model=..., input=messages, ...)`. Единый источник правды прод/тесты (OpenAI выключает
  `v1/prompts`: 03.06.2026 / 30.11.2026). Рекомендованный запуск — через Docker (см. `docs/LOCAL_PROMPTS.md`).
- **stored** (дефолт для обратной совместимости) — `platform.openai.com`, `responses.create(prompt={id,version})`.

Флаги у всех LLM-раннеров (через `core.add_prompt_source_args`): `--prompt-source {stored,local}`
(или env `QA_HARNESS_PROMPT_SOURCE`) · `--local-prompt-version vN` (пин версии; иначе `pointer.yaml active`
в самом пакете — так тестируют не-дефолтную версию, не трогая пакет) · `--prompts-path`/env
`PROMPTS_REPO_PATH` (ДЕВ-обходной путь к исходникам). По умолчанию берётся **установленный релиз** (как в
проде — wheel из GHCR-образа `ghcr.io/podbor/prompts`, установка — `docs/LOCAL_PROMPTS.md`); неявного
подхвата соседнего `../prompts` нет (иначе тестировали бы локальную копию вместо релиза). Резолв версии в пакете: `--local-prompt-version` > env
`<COMPONENT>_PROMPT_VERSION` > `pointer.yaml active`. Маппинг имён (`model.yaml` → директория в `prompts`)
— поле `local_component` в `model.yaml` (`first_touch`→`FIRST_TOUCH`, `screening_autofill`→
`screening_autofill_prompt` и т.п.; где не задан — identity). **Screening тестируется в local** (v51),
в отличие от eggplant-api. Фабрика `core.make_prompt_client` → `StoredPromptClient`|`LocalPromptClient`
(общий `.run()`); мультитёрн `ScreeningConversation` в local шлёт system как `instructions=` при
серверном `conversation=`. `meta.prompt_under_test.source` в отчёте = `stored|local`.
Тестирование — локальное: пакет `prompts` ставится в venv (wheel из GitHub Release / GHCR-образа —
`docs/LOCAL_PROMPTS.md`). Неявного подхвата соседнего `../prompts` нет (только явный `--prompts-path`).

## Вариативная генерация (движок `domain/generators`)
Помимо курируемых golden у раннеров есть режим `--generate` — вариативная LLM-генерация входов, чтобы
проверять робастность промпта на РАЗНЫХ входах (а не на фиксированном эталоне). Общий движок:
- `core` — `generate_valid(produce, validate, policy, fallback)`: цикл **produce → validate → retry → fallback**
  + трасса (`GenResult.source` = `llm|fallback|failed`, attempts, usage). Не знает домена — вызывающий
  замыкает контекст/клиент в `produce`. Исчерпание → `source=failed` (не бросает) → раннер в `errors`.
- `VariantSampler(seed)` — детерминированный поверхностный стиль (тон/объём) для разнообразия прогонов.
- **Три типа producer:** (1) адаптивный LLM-кандидат `CandidateAgent` (отвечает на реплики ассистента вживую
  — screening_scenarios/guardrails); (2) seeded-сообщение/диалог с известной меткой (классификаторы,
  autofill work_format, responsibilities/one_line — засеваем термины → `expect`); (3) генератор контекста
  (`VacancyGenerator` — first_touch).
- **fallback — опция per-runner:** включён там, где запасной вариант безвреден (screening_scenarios —
  канон-реплика); ВЫКЛючен для размеченных данных (классификаторы/autofill/responsibilities) — там лучше
  недодать (`errors`), чем влить мислейбл в метрику.
- **Три режима входа:** `--golden` (курируемый, детерминизм/CI) · `--generate` (вариативный) · `--offline`
  (replay). 3 модели разведены: генератор (`gpt-4.1-mini`) ≠ судья ≠ промпт-под-тестом. `--generate` при
  `temperature>0` НЕдетерминирован → в CI-гейт не годится (для гейта — golden).
- Раннеры с `--generate` (**весь флот**): screening_scenarios, screening_guardrails, screening_autofill,
  message_classifier, verdict_classifier, first_touch (+`--component first_touch_hh`), responsibilities_parser,
  one_line_search_query_builder, **sourcing_assistant** (LLM-кандидат + засеянные requirements с известными
  `expect_passed` → контракт + СЕМАНТИКА 0/1; backend НЕ нужен — кандидаты генерятся, а не ищутся).
- Констрейнты/персоны/словари — ДАННЫЕ: `tests/fixtures/generation/<runner>/*.yaml`,
  `TECH_VOCAB`/`SOFT_NOISE`/`DOMAINS` в `domain/generators`.

## Окружение
```bash
python3 -m venv .venv && source .venv/bin/activate     # WSL/Linux
pip install -e .[dev]                                   # qa_harness + dev-инструменты (import-linter и пр.)
lint-imports                                            # контракт qa_harness ⊥ app (pytest-тестов в репо нет)
```
`OPENAI_API_KEY` — всем LLM-раннерам. Для backend-поиска (extractor step2/3, будущие one_line/sourcing):
`AI_SEARCH_BASE_URL` + `AI_SEARCH_AUTH_TOKEN`. Backend тест-стенда: **`https://testsecond.hlebusheck.ru`**
(эндпоинт `/site/searchBool`), токен — **в теле** (флаг `--token-in-body`). `podbor.io/search` — это веб-UI, НЕ API.

### Ответ `/site/searchBool` — структура (НЕ только число)
Backend на `searchBool` возвращает JSON **`{count, profiles: [...]}`**: `count` — сколько кандидатов нашлось,
`profiles` — массив объектов-кандидатов (`about`, `skills`, `positions` и пр.), **из которых можно вытащить
самих кандидатов**, а не только их количество. Сколько `profiles` придёт, задаёт `limit` в payload:
- `limit=0` → только `count` (быстро, без профилей) — так работает `sourcing --count-only` и step3 у
  `extractor_agent`/`one_line` (им нужен лишь count как retrieval-инфо);
- `limit=N>0` → backend кладёт до N объектов в `profiles` — так достают РЕАЛЬНЫХ кандидатов.

`core`/`pipeline.call_backend_search_bool(...)` возвращает кортеж `(kind, status, attempts, count, error, json)`,
где **6-й элемент `json` — это полный ответ** (`{count, profiles}`); `count` отдаётся отдельно для удобства,
но профили берутся из `json["profiles"]`. Где это уже используется для извлечения кандидатов:
- **новая арх.:** `qa_harness.runners.sourcing_assistant` — `_process_online` (по CDM-entities) и `_process_search`
  (по реальным вакансиям через `--search`): `response["profiles"]` → `build_candidate_profile` → scoring;
- **легаси:** `app/sourcing_assistant_runner.py` → `_search_backend_candidates` (`backend_response.get("profiles")`).

## Грабли (важно!)
- **Большинство раннеров НЕ читают `.env`** — экспортируй в окружение (`set -a; source .env; set +a`).
- `prompt_id`/`version` — **только из `model.yaml`** (источник правды); env/CLI-override опционально,
  а `screening_scenarios_runner`/`screening_guardrails_runner` override игнорируют.
- **extractor step3: по умолчанию `limit=0` (только count).** Backend отдаёт `count` за ~11с, но
  тяжёлые `profiles` (limit>0) — минутами/таймаут. Не ставь большой `--step3-limit` без нужды.
- extractor: **качество промпта ≠ инфраструктура** — backend timeout/auth/http идут в `errors`, не в `failed`.
- Окружение машины: рабочий venv — Linux/WSL; если терминал MSYS/git-bash — гоняй через
  `wsl.exe -e bash -lc '... source .venv/bin/activate ...'`.
- Sourcing-раннеры (легаси) требуют в CDM `vacancy.extractor_entities`/`raw_vacancy`
  (`python app/enrich_cdm_with_extractor_entities.py`). `make_vacancies.py` их не генерит.
- `sourcing_assistant` (новый): профили (`limit>0`) — медленный путь, таймаутят ШИРОКИЕ вакансии (высокий
  count), не настройка. Триаж — `--count-only` (limit=0, ~сек) → потом полный прогон по узким. `SSLEOFError`/
  `Max retries` — транзиентный обрыв соединения (не таймаут), лечится повтором.
- chain-группы сценариев зашиты в коде ЛЕГАСИ `app/screening_scenarios_runner.py`. В новом
  `qa_harness.runners.screening_scenarios` их нет — LLM-судья (`ScenarioJudge`) против `expected_behavior`
  из CSV; онлайн гоняются только сценарии с примерами диалога кандидата (base CSV: 10/64, hh CSV: 0/50).

## Частые команды (новые раннеры)
```bash
python -m qa_harness.runners.message_classifier --offline
python -m qa_harness.runners.verdict_classifier --mode all --dialogs-per-verdict 3 --seed 42
python -m qa_harness.runners.extractor_agent --steps 1                      # тест промпта без backend
python -m qa_harness.runners.extractor_agent --steps 1,2,3 --token-in-body  # + count из backend
```

## Конвенции
- Всё (промпты, сценарии, golden, комментарии) — на русском. Сохраняй язык.
- JSON-отчёты и фикстуры — UTF-8 без BOM, `ensure_ascii=False`.
- Меняя версию промпта — правь `tests/tools/model.yaml`, не хардкодь.
- Новый код — в `qa_harness`, не в `app/`; новое не должно импортировать `app/` (проверяет `lint-imports`).

# План рефакторинга AI-Agents (v2 — параллельная сборка)

> Статус: предложение для ревью. Код раннеров пока не трогаем.
> Этот документ описывает **как** мигрировать. Схема отчётов — в [REPORT_SCHEMA.md](REPORT_SCHEMA.md).

## 0. TL;DR

Сейчас репозиторий — это ~17 копий одного скелета раннера без общего дома (~22 900 строк).
Цель — вынести инфраструктуру и доменную логику в один устанавливаемый пакет, схлопнуть
дублирование, унифицировать отчёты. **Ключевое ограничение: старый код не редактируем и не
удаляем in-place — строим новое рядом и переключаемся одним cutover-коммитом.**

Решения:
- **Один пакет `src/qa_harness/`** (src-layout), не россыпь top-level папок. «Несколько папок» =
  вложенные подпакеты `core/ pipeline/ domain/ runners/` под одним корнем.
- **Миграция вертикальными срезами** (один раннер целиком → следующий), а не strangler-in-place.
- **Парити-харнесс на кассетах (VCR)** как ворота каждого среза — вместо иллюзорного snapshot-эталона.
- `app/` живёт нетронутым до финального cutover-PR (мгновенный откат, живой оракул для парити).

## 1. Диагноз (проверено по коду)

| Что | Факт |
|---|---|
| Всего строк в `app/` | ~22 900 |
| Два скрининг-раннера | `screening_scenarios_runner.py` 4350 + `..._hh_runner.py` 4084 (≈70–80% перекрытия) |
| usage-триада `_blank/_extract/_accumulate_usage` | в 10+ файлах |
| `OpenAI(api_key=…)` | ~22 сайта; запись отчёта — ~35 сайтов / 15 файлов |
| Sibling-импорт примитивов из CLI-раннера | `one_line_…:14`, `sourcing_…:21`, `enrich_…:14` тянут `from extractor_agent_runner import …` |
| `sys.path.append` хак | `first_touch_runner.py` (был; устранён при выносе telegram-генератора) |
| Мёртвый код в HH | дубль-определения `_assistant_reply_has_context_leak` (L2764 мёртв/L2846 живой), `_reply_presents_agency…` (L2791/L2858, регэкспы расходятся) |
| Отчёты: BOM | `first_touch_hh`, `message_classifier`, `screening_autofill`, `verdict_classifier` пишут UTF-8 **с BOM** (`utf-8-sig`), остальные — без |
| Отчёты | 17 несовместимых JSON-схем; «провалы» названы по-разному (`mismatches`/`failures`/`failed_turns`/`execution_errors`) |
| `seed` | в `model.yaml` у всех блоков `seed: 1234` (НЕ null), но в `responses.create(...)` **не пробрасывается** → генерация недетерминирована |

Вывод: команда хочет общие примитивы (доказано sibling-импортами), но им негде жить.

## 2. Целевая структура: один пакет `src/qa_harness/`

Из трёх вариантов размещения нового кода:
1. **Один пакет, src-layout** — **ВЫБРАН** (best practice PyPA, истинно недеструктивно, чистый cutover).
2. Россыпь top-level папок `core/ generators/ judges/` рядом с `app/` — **отклонён**: пересоздаёт
   cwd-зависимую хрупкость (импорт работает только потому, что repo-root в sys.path) и загрязняет
   глобальное пространство генерик-именами.
3. Strangler in-place (план v1) — **отклонён**: противоречит ограничению пользователя.

```
ai-agents/
├── pyproject.toml                # НОВЫЙ: декларирует ОБА дерева на время миграции
├── requirements.txt              # НЕТРОНУТ (старый workflow продолжает работать)
├── app/                          # СТАРЫЙ — НЕТРОНУТ, по-прежнему `python -m app.<runner>`
├── adapters/                     # СТАРЫЙ общий модуль — НЕТРОНУТ
├── src/qa_harness/               # НОВЫЙ — единственное, что растёт
│   ├── core/                     # ИНФРАСТРУКТУРА only
│   │   ├── usage.py   jsonio.py  config.py  llm_client.py
│   │   ├── reporting.py          # ReportBuilder → two-file schema, utf-8 без BOM
│   │   ├── metrics.py  cli.py
│   ├── pipeline/                 # extractor step1/2/3, backend_client, payload, contract
│   ├── domain/                   # ВСЁ рекрутинговое (НЕ в core!)
│   │   ├── text/                 # detectors, города, salary-shorthand (mojibake починен)
│   │   ├── judge/                # Judge-протокол, PerTurnLLMJudge, BatchedLLMJudge, LabelJudge, ContractJudge
│   │   ├── generators/           # base + message_gen + dialogue_gen + scenario_gen
│   │   ├── screening/            # движок (variant = std | hh)
│   │   └── first_touch/          # checks.py + judge.py (только plumbing)
│   ├── runners/                  # ТОНКИЕ раннеры (~120–200 строк)
│   └── legacy_bridge.py          # единственное место знания о старых артефактах; удаляется на cutover
├── parity/                       # парити-харнесс на кассетах (см. §5)
├── config/model.yaml             # канонический дом (single-source, см. P0-3)
└── tests/
    ├── fixtures/cdm/{std,hh}/    # ОБЩИЕ read-only
    ├── reports/                  # старые 17 shape'ов — старые раннеры пишут сюда
    └── reports_v2/               # НОВЫЙ two-file schema — новые пишут сюда (раздельно)
```

### Граница core ↔ domain (поправка критика)
`core/` = **только инфраструктура** (usage/jsonio/config/reporting/llm_client/cli/metrics).
Всё рекрутинговое (детекторы work_format, префиксы диалога, метки классификатора, screening-движок,
критерии судьи) — в `domain/`. Иначе «ядро» окажется на 70% про русский рекрутинг.

## 3. ⚠️ Префлайт-блокеры (закрыть ДО первой строки раннера)

Эти пункты вскрыл адверсариальный разбор плана; без них «низкий риск» — самообман.

- **P0-1. Проверить, что `pip install -e .` (src-layout) НЕ ломает старые раннеры.**
  Bare-импорты в `app/` (`from extractor_agent_runner import …`) и `python -m app.x` работают
  сейчас за счёт cwd/script-режима. Резолюция: **старые раннеры продолжают запускаться как сегодня
  (`python app/x.py` из корня, без install)**; editable-install нужен только для нового `qa_harness`.
  Старый код не обязан быть импортируемым через установленный пакет. Проверить это реально, до кода.
- **P0-2. Заморозить `app/` контрактом, а не надеждой.** Ветка активно правит раннеры прямо сейчас.
  Политика на время миграции: любой коммит в `app/` требует `parity record` + `verify-old`
  затронутого раннера в том же PR. Иначе «нетронутый оракул» — фикция, и парити доказывает
  эквивалентность мёртвому baseline.
- **P0-3. Single-source `model.yaml`.** Никаких двух физических копий. Либо `config/model.yaml` —
  единственный файл (а `core/config.py` всегда читает его), либо CI-гейт «`tests/tools/model.yaml`
  и `config/model.yaml` побайтово равны». Иначе бамп `prompt_version` в одной копии рассинхронит парити.
- **P0-4. Развести «парити» от «фикс-багов».** mojibake-детекторы (`_COMMON_CITIES`,
  `_SALARY_SHORTHAND_SUFFIX`) — это баги матчинга: новый код должен давать ДРУГОЙ (правильный)
  результат. Это противоречит «новый == старый». Нужен явный **allow-list ожидаемых дивергенций**
  в `parity/families.py`: «парити = совпадение везде, КРОМЕ зарегистрированных расхождений».
- **P0-5. Проектировать `ReportBuilder`/схему против двух непохожих раннеров сразу** (classifier +
  extractor), а не только простейшего. Иначе схема забетонируется на label-match и переписывается
  на срезе 3 (`stages[]`), обнуляя уже «закрытые» парити-ворота срезов 1–2.

## 4. Абстракции (то, о чём просил пользователь: вынести генератор и судью)

### Генератор — общая база, конкретика в 3 методах
```python
class Generator(ABC):
    def __init__(self, client, *, model=GEN_MODEL, max_retries=1): ...
    @abstractmethod
    def instruction(self, spec) -> str: ...
    @abstractmethod
    def payload(self, spec) -> str: ...
    @abstractmethod
    def parse(self, text) -> list: ...
    def generate_with_fallback(self, spec, *, want, fallback, validate): ...
```
Реализации: `MessageGenerator`, `DialogueGenerator`, `ScenarioGenerator`. Клиент/usage/retry/fallback — в базе.

### Судья — протокол + ДВА класса (поправка критика: batched ≠ per-turn)
Подтверждено по коду (`evaluate_turn` vs `_evaluate_chain_case` L2662): per-turn и batched —
**две стратегии** с разным payload-контрактом, инструкцией и парсингом, а не «один флаг».
```python
class Judge(Protocol):
    def evaluate(self, case, reply, *, criterion, context) -> Verdict: ...
```
- `PerTurnLLMJudge`, `BatchedLLMJudge` (HH), `LabelJudge` (классификаторы), `ContractJudge` (поиск).
- `criterion` (`expected_behavior`) — **обязательный аргумент**, не опциональное поле отчёта: судья
  физически не может оценить кейс без критерия, и критерий автоматически протекает в `cases.json`.
- Детерминированные правила (`enforce_*`) — composable `Check`'и поверх вердикта (`CompositeJudge`).

### Клиент — кэш по конфигу, не singleton
`get_client()` кэширует по `(base_url, timeout)`; разделить `StoredPromptClient` (prompt-under-test)
и `ModelClient` (gen/eval). Это нужно для подмены fake-клиента в offline/replay/парити.

## 5. Парити-харнесс — ворота миграции

Детерминизм нельзя получить от модели — его получают от записи (VCR/cassette).
- **Перехват на HTTP-транспорте (`vcrpy`), не на SDK.** Подтверждено: extractor/one_line ходят сырым
  `requests.post` в обход SDK + backend `/site/searchBool`; мок `openai.OpenAI` их утечёт в сеть.
- **Три фазы:** `record` (старый код, единственный сетевой вызов) → `verify-old` (старый код из
  кассеты обязан воспроизвести нормализованный baseline) → `verify-new` (новый раннер под
  role-sequence matcher, структурный diff vs baseline; любой незаписанный запрос = hard fail).
- **Нормализация** (`parity/normalize.py`): стрип `run_id/*_at/duration/git_commit/пути`;
  `token_usage` → `"<usage>"`; `utf-8-sig` (лечит BOM); сортировка ключей и кейсов; разреженная
  confusion_matrix.
- **«Парити» per family:**

  | Семейство | Точное совпадение | Допустимый дрейф |
  |---|---|---|
  | classifier | predicted_label, passed, accuracy, confusion_matrix | порядок, raw text, usage |
  | judge-based | passed/score/reason_codes per case+turn | transcript-структура, комменты |
  | guardrails | per-turn флаги + used_heuristics | repeated_topics текст |
  | search-pipeline | passed, contract-исход, **step2 payload**, backend-классификация | avg_candidates, usage |
  | sourcing | **per-candidate** passed | агрегаты |

- **Оговорка:** парити при batched-vs-per-turn судье **меняет число вызовов модели намеренно**
  (batched = 1 на цепочку, per-turn = N) — role-sequence matcher должен это допускать через allow-list
  (P0-4), иначе судью нельзя извлечь. Полнота кассеты для 4000-строчных screening-раннеров при
  `--limit 3-5` не гарантирована — `verify-old` это ловит.

```
python -m parity.parity record     <runner> <scenario>   # старый код, сеть, 1 раз
python -m parity.parity verify-old <runner> <scenario>   # кассета валидна?
python -m parity.parity verify-new <runner> <scenario>   # ПАРИТИ — CI-gate перед мержем
python -m parity.parity accept     <runner> <scenario>   # осознанно перезаписать baseline
```

## 6. Порядок работ (вертикальные срезы, не breadth-first)

- **P0 — фундамент:** `pyproject.toml` (оба пакета), `parity/` каркас + `normalize.py` (пишется
  ПЕРВЫМ), vcrpy-обвязка, import-linter контракт `qa_harness ⊥ app`. Закрыть префлайт-блокеры §3.
- **Срез 1 (эталон) — `message_classifier`:** простейший, без LLM-судьи. Строит каркас `core/` +
  `domain/judge/label_judge` + `domain/generators/` + `reporting` (два файла) + offline/replay/
  differential + JSON-схема. Спроектирован сразу с оглядкой на extractor (P0-5).
- **Срез 2 — `verdict_classifier`:** почти идентичен → валидирует переиспользуемость каркаса.
- **Срез 3 — `extractor_agent`:** достраивает `pipeline/` и `ContractJudge`; вскрывает `stages[]`
  РАНО. Даёт дом для sibling-импортов.
- **Срез 4 — `one_line` / `sourcing` / `enrich`:** их новые версии импортируют из `qa_harness.pipeline`
  с рождения (анти-паттерн не возникает). `sourcing` валидирует `subjects[]` (per-candidate).
- **Срез 5 — `first_touch`:** достраивает `domain/first_touch/` (~300 строк plumbing). Три раннера
  НЕ сливаются в один (tg/hh/event разнородны) — общий только plumbing.
- **Срез 6–7 — screening STD, затем HH как второй variant того же движка.** ⚠️ «один движок, два
  variant» — гипотеза: два 4000-строчных файла с раздельными `_evaluate_chain_case` могли разойтись
  по существу. Сначала доказать общность движка на STD, потом проверять обобщение на HH.
- **Cutover:** `git rm -r app/ adapters/`, убрать legacy из `pyproject`, переключить config на
  `config/model.yaml`, удалить `legacy_bridge.py`, обновить README/CLAUDE.md на console-scripts.
  Один ревьюабельный коммит; регресс → откат одного коммита.

## 7. Честные tradeoffs

- **Ease of use временно УХУДШАЕТСЯ.** На срок миграции: два namespace (`app.*`/`qa_harness.*`),
  две команды запуска, две папки отчётов, новый `pip install -e .`, packaging-слой, import-linter,
  vcrpy, parity-CLI. Окупается на cutover — но cutover в конце многонедельной очереди срезов.
- **Дрейф `app/` во время миграции** — главный новый риск parallel-build (P0-2).
- **Два формата отчётов одновременно** (`reports/` + `reports_v2/`); `normalize.py` обязан понимать оба.
- **Разовая стоимость packaging** до переноса кода (репо это пока избегал).
- **Парити доказывает эквивалентность только при зафиксированных ответах модели** — не валидирует
  качество промпта и не ловит расхождения на не-записанных входах. `--limit 3-5` подбирается под покрытие код-путей.

## 8. Ожидаемый эффект

| Метрика | До | После (оценка) |
|---|---|---|
| Строк всего | ~22 900 | ~10 000–12 000 |
| Два screening-файла | 8 434 | ~1 650 (движок) + 150 (раннер) |
| usage-триада | 10 файлов | 1 (`core/usage.py`) |
| `OpenAI(api_key=…)` | 22 сайта | 1 (`get_client`) |
| Запись отчёта | 35 сайтов | 1 (`ReportBuilder`) |
| Реализаций судьи | 4 несовместимых | 1 протокол + N классов |
| Юнит-тестируемость примитивов | нет (god-функции) | да (`core/` под pytest) |

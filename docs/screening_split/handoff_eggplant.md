# Порт policy-ядра в eggplant-api — бриф для новой сессии

Документ самодостаточный: его можно вставить в чистый чат целиком и начать писать код. Задача, под
которую он написан — **перенести новое ядро скрининга (`policy`) в `eggplant-api`, HH-канал**.

Состояние на 31.08.2026. Механику самого ядра документ не пересказывает: она в
[mechanics_policy.html](mechanics_policy.html) (порядок хода, таблица правил, бюджеты, гарды, реестр)
и в [handoff_two_engines.md](handoff_two_engines.md) (старый движок против нового, «известная ошибка →
повторится или нет»). Состояние репозиториев и веток — в [../repos.md](../repos.md). Открытые пункты —
в [plan_cross_repo.md](plan_cross_repo.md).

---

## 0. Правила работы, прежде чем трогать код

`eggplant-api` — **продуктовый** репозиторий, а `ai-agents` — штаб. Отсюда следует, и это не
формальности:

1. **Сначала ветка от свежего `master`.** В репозитории идёт чужая работа
   (`fix/screening-session-rollback`) — от неё не ветвиться. Коммит — только по явной просьбе.
2. **`git push` и открытие PR запрещены правами** (`.claude/settings.local.json` → `permissions.deny`).
   Пушит человек. Обходить запрет нельзя.
3. **Конвенции — ИХ.** У репозитория свой `CLAUDE.md` (архитектурный курс «тонкий прокси над HH»),
   `TASKS.md`, `TECH_DEBT.md`; для его файлов они главнее правил штаба. Коммиты
   `PO-#### англ. суть (#PR)`. Русский чейнджлог-коммит и «pytest не держим» — это правила `ai-agents`,
   на eggplant они НЕ распространяются: там pytest есть и он гейт.
4. **Наши заметки в их репозиторий не кладём** и ссылок оттуда сюда не ставим. Знание живёт здесь.
5. **pytest, docker-compose и Alembic гоняются из `eggplant-api`**, не из штаба. Отсюда гоняются
   только тесты промптов (раннеры `qa_harness`).
6. В репозитории лежат два untracked-файла (`TODO_salary_reask_cap.md`,
   `docker-compose.override.yml`) — не наши, содержимое первого учтено в плане.

---

## 1. Что переносим и откуда

**Исходник переноса — hh-ядро харнесса, а не tg-ядро.** Оно уже канальное: мультиформат, ветка
разъездного формата, ключи отсева HH. Путь: `ai-agents/src/qa_harness/domain/screening_split_hh/policy/`.

| модуль (харнесс, hh) | строк | что делает | канальная дельта к tg |
|---|---|---|---|
| `core.py` | 551 | `decide(state, observation, message, ctx) -> TurnPlan` — **чистая функция хода**: ни сети, ни стора, ни LLM | повестка из ЧЕТЫРЁХ пунктов (зарплата → присутственный формат → разъездной → доп-вопросы), выбор формата для вопроса, накопление ответов по форматам |
| `rules.py` | 138 | таблица правил R1–R11, первое сработавшее выигрывает | R5 `KO_LOCATION_GEO` · **R5a `KO_LOCATION`** (отказ про МЕСТО) · R6 `KO_FORMAT` по мультиформату · R10 без ветки источника контакта |
| `observation.py` | 89 | контракт и мягкая валидация ответа Наблюдателя | нет сигнала `contact_source` (21 код вместо 22); `facts.formats_ready = [{format, ready}]` вместо одного `format_ready` |
| `formats.py` | 80 | мультиформат чистыми функциями: «подтвердил хотя бы один», «отказался от всех», «о каком формате спрашивать следующим» | модуля в tg нет вовсе |
| `reasons.py` | 132 | `reason_code → {text, terminal, author}` + три инварианта ПРИ ИМПОРТЕ | `KO_FORMAT`/`KO_LOCATION`/`KO_LOCATION_GEO` вместо `KO_FORMAT_OFFICE/_HYBRID/_NOCITY` и `KO_GEO`; без `REPLY_CONTACT_SOURCE` |
| `budgets.py` | 43 | пороги данными: событийные, reask, stall | без `contact_source`, зато с `field_work`; значения те же |
| `context.py` | 56 | контекст для Наблюдателя (**без вилки**), `allowed_formats_of`, `has_geo_restriction` | свои лейблы строк |
| `observer.py` | 87 | «уши»: один вызов LLM → `Observation`, до 3 попыток | берёт hh-проекцию состояния и hh-разбор |
| `engine.py` | 192 | тонкий оркестратор: стор → Наблюдатель → `decide` → речь → гарды → стор | без ленивой миграции, без `contact_source`, вилка приходит типизированной |
| `__init__.py` | 54 | публичный контракт пакета | — |

**Канало-независимое hh-ядро импортирует из tg-ядра** — в eggplant эти файлы приедут как копии:

```
policy/budgets.py     ← Budget (класс порога)
policy/context.py     ← _GEO_MARKERS
policy/core.py        ← salary (модуль), общие части ядра, geo.relocation_pointless
policy/engine.py      ← guards.GuardSpec/_urls/apply_guards, _compat_decision (только для QA-трассы)
policy/observation.py ← TERMINAL_PRIORITY, Signal, общий разбор
policy/reasons.py     ← Reason, CODE/MODEL/NON_SCRIPT
policy/rules.py       ← R1, R2, R3 (+R3a/R3b), R4, R8, R9, R11, geo.same_city
```

Значит в `eggplant-api` едут ещё и: `policy/guards.py` (249), `policy/gates.py` (95),
`policy/geo.py` (39), плюс **зарплатный модуль** `salary.py` + `salary_rules.py` — его в eggplant нет
вовсе (см. §4).

**Эталон обвязки — ветка PR в `tgApi`** (`feat/screening-policy-engine`): там тот же порт уже сделан
для Mongo и синхронного тракта. Состав (пути от `tgApi/app/common/screening/`):

| что | файлы |
|---|---|
| ядро | `policy/`: `core.py` · `rules.py` · `guards.py` · `observation.py` · `reasons.py` · `migration.py` · `context.py` · `gates.py` · `geo.py` · `budgets.py` · `__init__.py` |
| зарплата | `salary.py` · `salary_rules.py` |
| роли | `ScreeningObserver.py` (пинит `screening_analyzer` **v3**) · `ScreeningInterviewer.py` (stateless) |
| оркестратор | `ScreeningSplitEngine.py` — 378 → 139 строк |
| хранилище | `screening_repository.py` (+`salary_band`, `salary_claims`, `schema`, `save_migrated`) · `screening_state.py` |
| удалено | `ScreeningAnalyzerAssistant.py`, `screening_scripts.py` — целиком |

Два решения оттуда, которые в eggplant поедут вместе с ядром:

- восстановлен `REPLY_CONTACT_SOURCE_EMPTY` (в hh неактуально — источника контакта в канале нет);
- **в файлах tg-PR нет ни одного комментария и докстринга** — снято по требованию владельца того репо.
  Для `eggplant-api` это правило **не действует автоматически**: там своя плотность комментариев,
  решать отдельно (по умолчанию — как в соседних файлах их репозитория).

---

## 2. Куда переносим — карта `eggplant-api` пофайлово

Скрининг: `app/assistants/screening/` — 549 строк против ~1300 у нового ядра.

| файл | строк | что там сейчас | что делать |
|---|---|---|---|
| `engine.py` | 174 | оркестратор, **async**. `run_turn(dialogue, message) -> (text, ended)`. Внутри: `EVENT_STOP`, `NO_PROGRESS_CAP=4`, `REASK_CAP=2`, `_force_by_counters`, `_apply_reask_cap` (с ВТОРЫМ вызовом Аналитика), `_render`, `_speak` | **переписать целиком** по образцу `policy/engine.py`: `run_turn` = загрузить → наблюдать → `decide()` → сказать → гарды → сохранить. Все счётчики и капы уезжают в ядро (`budgets.py`), второй вызов Аналитика исчезает структурно |
| `assistants.py` | 129 | `ScreeningAnalyzerAssistant` (валидатор `Decision` на 6 полей) + `ScreeningInterviewer` (**stateful**, через `conversation_id`) | Аналитик → **Наблюдатель** (`Observation`, мягкая валидация, до 3 попыток); Интервьюер становится **stateless**: на вход `instruction` + сообщение + `seed` + `last_sent`, истории не читает |
| `state.py` | 113 | `init_state`/`apply_updates`/`progress_signature`/`is_complete`/`find_question` | дописать ключи ядра (§5), `apply_updates` — ветку `relocation_ready`, `progress_signature` — `formats` |
| `scripts.py` | 81 | реестр текстов + `is_known`/`is_terminal`/`render_script`/`fallback_text` | **заменить на `policy/reasons.py`** (реестр с автором и инвариантами при импорте). `is_terminal` по префиксу имени уходит — терминальность берётся из реестра |
| `context.py` | 52 | `build_context` (**вилка строкой внутри**, «(НЕ РАСКРЫВАТЬ!)»), `build_interviewer_seed`, `build_greeting` | из контекста Наблюдателя **убрать строку вилки** (П4: секрет не кладут туда, где его потом запрещают), вилку передавать типизированной. `build_greeting`/`build_interviewer_seed` остаются как есть |

Обвязка, которую нельзя сломать:

- **вызов движка** — `app/assistants/service.py`:
  - `create_screening_conversation` → `_create_split_conversation`: создаёт conversation у
    Интервьюера ради ID, пишет строку `ScreeningDialogue` c `init_state(allowed_formats,
    screening_questions)`, `context`, `location`, `greeting`;
  - `generate_screening_message` → `run_turn(dialogue, message)`; **наличие строки в БД и означает
    split-движок** (легаси-ветка `_generate_screening_message_legacy` — для диалогов без строки);
  - **двухфазная отправка (outbox):** `get_pending_screening_message` / `confirm_screening_send`.
    `run_turn` кладёт текст в `dialogue.pending_text` и **не коммитит**; коммит — за вызывающим;
  - флаг `config.SCREENING_ENGINE == SPLIT_ENGINE` решает, каким движком СОЗДАВАТЬ новые диалоги.
- **хранилище** — Postgres, `app/assistants/models.py::ScreeningDialogue` (таблица
  `screening_dialogues`): `openai_conversation_id` (уникальный ключ), `state` — **JSONB**, плюс
  `context`, `location`, `greeting`, `pending_text`, `finished`. Доступ —
  `app/assistants/repository.py`: `create` / `load` / `save_state` / `clear_pending`.
- **промпты** — не хардкод, а конфиг: `config.SCREENING_ANALYZER_PROMPT_NAME` /
  `SCREENING_INTERVIEWER_PROMPT_NAME`.
- **миграции — Alembic** (`app/migrations/versions/`, `alembic.ini`). Новое поле уровня КОЛОНКИ
  требует ревизии; поля внутри `state` (JSONB) — нет.
- **тесты — гейт:** `app/tests/assistants/test_screening_engine.py` (367),
  `test_screening_state.py` (146), `test_screening.py` (96), `test_screening_scripts.py` (53).

---

## 3. Промпт Наблюдателя: чего ещё нет

`screening_analyzer_hh/v3` **написан, но лежит в репозитории `prompts` untracked** и в релиз 1.2.2
не попал. Контракт `Observation` тот же, что в tg (7 полей), с двумя канальными отличиями; остальное,
включая блок `salary_claim`, перенесено из `screening_analyzer/v3` дословно.

- **нет сигнала `contact_source`** — 21 код вместо 22;
- **готовность к формату по КОНКРЕТНЫМ форматам:** `facts.formats_ready = [{format, ready}]`.
  «Подходит хотя бы один» — вывод КОДА по `allowed_formats`;
- **новое поле STATE `format_asked`** (пишет код): формат, о котором спросили прошлым ходом. По нему
  Наблюдатель относит короткое «да»/«нет» к конкретному формату, а не вычитывает из прозы
  `last_asked`. Без него мультиформат держался бы на догадке модели.

Зарплатных правил v2 (диапазон по верхней границе, уточнять голое «260», gross с вилкой не
сравнивать) в v3 НЕТ: они противоречили решениям Д8/Д11/Д13, а считает теперь код. Тело — 24,7 тыс.
символов против 41,8 тыс. у v2. Вилки в контексте больше нет.

**Порядок между репозиториями: сначала КОД в master, потом релиз промпта.** Код терпит отсутствие
поля, промпт без кода отдавал бы наблюдение в пустоту. Тела выпущенных версий не правим: нужен другой
текст — это v4.

---

## 4. Зарплата: модуля нет вовсе

Сегодня в hh вилка живёт **строкой** внутри `context`, решение об отсеве целиком за моделью, а
`currency` из HH не читается — вилка в тенге сравнивается как рублёвая (это открытый пункт P11).

Переносим из харнесса (`domain/screening_split/`): `salary.py` (156) + `salary_rules.py` (22) —
курсы, нормо-часы, шкала НДФЛ — и `policy/gates.py` (односторонние гейты). Что даёт ядро:

- `claim_status` → `normalize` (масштаб → курс → период → gross→net) → `compare_with_band`;
- отказ **только если НИЖНЯЯ граница ожиданий выше нашего максимума**;
- вилка типизирована `{min, max, currency}` и приводится к рублям (`DecideContext.band_currency`);
- ключа `salary` в контракте Наблюдателя нет — пункт закрывает КОД после сравнения;
- аудит claim: в tg это `store.log_salary_claim`; в eggplant решить, куда писать (лог или таблица) —
  без него «сумму не распознали» не отличить от «распознали и не отсеяли».

`currency` берётся из `hh_vacancy_data["salary"]`. **Дотащить это чтение обязательно**, иначе валютный
дефект переживёт порт. Где вилка хранится и что делать со старыми строками — §5.1 и §5.2.

---

## 5. Состояние: точная дельта и миграция

Текущий `init_state` (eggplant) против нужного ядру. Совпадают: `salary`, `format_check`,
`field_work_check`, `candidate_city`, `allowed_formats`, `questions[]`, `counters`, `last_asked`,
`last_asking`, `salary_reasks`, `format_reasks`, `field_work_reasks`, `no_progress`.

| ключ | назначение | откуда |
|---|---|---|
| `formats` | `{'ON_SITE': 'yes'\|'no', …}` — ответы ПО ФОРМАТАМ, накопленные за диалог. `Observation` отдаёт только сказанное на этом ходе, а «отказался от всех допустимых» считается по накопленному | **добавить** |
| `format_asked` | формат последнего заданного вопроса; по нему Наблюдатель относит «да»/«нет» | **добавить** |
| `relocation_ready` | `'yes'\|'no'\|None`. Немонотонно: кандидат вправе передумать | **добавить** |
| `questions_intro_sent` | вводная перед первым доп-вопросом сказана (правка Б1) | **добавить** |
| `last_sent` | что реально ушло кандидату прошлым ходом — нужен stateless-Интервьюеру, иначе переспрос повторится дословно | **добавить** |
| `greeted` | приветствие приклеено к первому сообщению | **есть в eggplant, в харнессе НЕТ** — не потерять при переносе |

`apply_updates` дописать: ветку `relocation_ready` (`yes`/`no`, последнее слово кандидата).
`progress_signature` дописать: `tuple(sorted(state["formats"].items()))` — ответ про конкретный формат
это прогресс, даже если проверка не закрылась.

### 5.1. Вилка: колонка `salary_band` (решение принято)

**Решение Р15:** вилка хранится снапшотом в строке диалога, **одной колонкой `salary_band` (JSONB)** —
как поле документа в tgApi. Вариант «собирать на лету из `hh_vacancy_data`» отклонён, хотя технически
дешевле: `KO_SALARY` — отказ человеку и обязан воспроизводиться по трассе, а скрининг живёт днями и
тикается воркером, который каждый тик заново тянет вакансию. Обоснование целиком —
[decisions_rearchitecture.md](decisions_rearchitecture.md), Р15.

- **Alembic-ревизия нужна** (одна): `salary_band JSONB NOT NULL DEFAULT '{}'::jsonb` в
  `screening_dialogues`. Именно объект, а не три скалярных колонки: вилка нигде не фильтруется, читается
  целиком в `DecideContext`, а симметрия трёх портов дороже стилистики.
- **Цена JSONB — проверки в КОДЕ, в одной точке чтения:** `currency` дефолтится в `RUB` **явно** (HH
  валюту может не отдать, и молчаливое «наверное рубли» — это ×80 в чужой валюте), `min`/`max`
  приводятся к `int | None`. Больше нигде вилку не трогать.
- **Заполняется при создании диалога** — `_create_split_conversation` уже получает `min_salary` /
  `max_salary`; добавить `currency` из `(hh_vacancy_data.get("salary") or {}).get("currency")`
  (`app/candidates/tasks/process_screening.py:214`) — сегодня это поле не читает никто, это и есть
  незакрытый P11.

### 5.2. Миграция существующих строк

Прод работает, диалоги, начатые старым движком, придут в новое ядро. Порядок как в tg: лениво при
первой загрузке, признак — ключ версии схемы внутри `state`.

1. **вилку разобрать из строки `context` ДО её вырезания.** Формат тот же, что в tg:
   `Зарплатная вилка: от X до Y рублей (НЕ РАСКРЫВАТЬ!)` (`context.py::salary_display`);
2. **бэкфилл обязан быть ГРОМКИМ.** Сколько строк обработано и у скольких вилку разобрать не удалось —
   числом в лог или метрику (в tg это `band_unparsed` в отчёте миграции);
3. **неразобранная вилка — не «проходит», а непригодная строка.** При пустой вилке у диалога,
   созданного до выката, ход отдаёт `REPLY_FALLBACK` и **не** помечает диалог завершённым: застрявший
   диалог видно, пропущенного кандидата нет. Сегодняшнее поведение (`compare_with_band` без вилки
   всегда «проходит») — тихий отказ защиты, самая дорогая ошибка бэкфилла;
4. **ключи `state` добить дефолтами** — это JSONB, делается КОДОМ при загрузке, ревизии не требует;
5. **`last_asking` обнулить**: у поля сменился автор (писала модель, станет писать код). Сравнение
   нового фокуса со старым значением начислит переспрос, которого не было.

---

## 6. Async: где ядро остаётся синхронным

`decide()` — чистая функция и остаётся **синхронной**: в ней нет ни сети, ни стора. Async нужен
только вокруг:

```
async def run_turn(dialogue, message) -> tuple[str, bool]:
    if dialogue.finished: return "", True
    state = dialogue.state
    ctx = DecideContext(band_min=…, band_max=…, band_currency=…,
                        location=dialogue.location, has_geo_restriction=…)
    observation, failed = await observe(dialogue.context, state, message)   # ← await, 1 вызов LLM
    plan = decide(state, observation, message, ctx, analyzer_failed=failed) # ← sync, без сети
    text  = await speak(plan, message, ctx, state.get("last_sent"))         # ← await, 0 или 1 вызов
    …guards, greeting, last_sent…
    repository.save_state(dialogue, plan.state_next, finished=plan.end, pending_text=text or None)
    return text, plan.end
```

**Вызовов LLM за ход: 0** (диалог закрыт) · **1** (ход-скрипт) · **2** (ход с вопросом). Третьего не
бывает структурно — второго вызова Наблюдателя в движке нет, вместе с ним уходит `_apply_reask_cap`
с его `rerun`.

**Две вещи, которых в харнессе нет, потому что там движок держит вакансию в памяти** (`self._vacancy`),
а в eggplant `run_turn` видит только строку БД:

- `has_geo_restriction` — в харнессе считается по маркерам в «Локации» и «Описании вакансии»;
- канонический URL для гарда G7 (подмена выдуманной ссылки) — в харнессе берётся из описания вакансии.

Оба поля есть в `dialogue.context` (там строки «Локация:» и «Описание вакансии:»). **Рекомендация:**
считать их из `context`, а не заводить новые колонки — контекст и так лежит в строке и переживает
рестарты. Альтернатива (колонки) требует Alembic и синхронизации с вакансией.

---

## 7. Что нельзя потерять — канальные отличия

| | tgApi (эталон переноса) | eggplant-api |
|---|---|---|
| исполнение | синхронное | **async** во всём тракте |
| хранилище | Mongo, схема по факту | Postgres + JSONB, колонки через Alembic |
| отправка | сразу текстом из хода | **outbox**: `pending_text` → `confirm_screening_send`; `run_turn` НЕ коммитит |
| приветствие | нет | `greeting` приклеивается к первому сообщению, флаг `state.greeted` |
| повестка | зарплата, формат, доп-вопросы | **плюс `field_work_check`**; `allowed_formats` в state (`REMOTE` среди допустимых снимает проверку присутственного) |
| ключи отказа | `KO_FORMAT_OFFICE/_HYBRID/_NOCITY`, `KO_GEO` | `KO_FORMAT`, `KO_LOCATION`, **`KO_LOCATION_GEO`** |
| источник контакта | событие `contact_source` + скрипт | **нет ни события, ни скрипта** — ни в `EVENT_STOP`, ни в `COUNTER_KEYS` |
| лимит переспросов | по пунктам раздельно, порог 3 (новая семантика) | `REASK_CAP = 2`, **общий**, старая семантика «сверяем ДО инкремента» → **привести к 3 по новой** (решение Р3), иначе пороги каналов разъедутся |
| зарплата | типизированная вилка + гейты | **модуля нет** (§4) |
| промпт Наблюдателя | `screening_analyzer` v3, в выпущенном пакете | `screening_analyzer_hh` v3, **untracked, не выпущен** |
| промпт Интервьюера | `screening_interviewer` v2 | `screening_interviewer_hh` v1 |

Четыре решения мультиформата, которых в tg нет и которые приезжают вместе с hh-ядром:

- **отказ от одного формата — не отсев.** Пока среди допустимых есть формат, о котором кандидат не
  высказался, код спрашивает про него («в офис не готов» при `[ON_SITE, HYBRID]` → вопрос про
  гибрид). `KO_FORMAT` — только когда отказ получен по ВСЕМ допустимым;
- **смена формата в вопросе не считается переспросом.** Кап тикает по паре
  (`last_asking`, `format_asked`): иначе кандидат, честно ответивший про каждый формат, сжигал бы
  бюджет собственными ответами;
- **`KO_LOCATION` против `KO_FORMAT` выбирает код** по смыслу отказа: отказ переезжать при известном
  городе, отличном от локации вакансии → `KO_LOCATION`; отказ от самих форматов → `KO_FORMAT`.
  Радиусов («Москва + 100 км») код не знает: нужен ЯВНЫЙ отказ от переезда;
- **код САМ спрашивает про переезд**, когда город известен и не совпадает с локацией. Без этого
  вопроса `relocation_ready` приходил только самотёком, `KO_LOCATION` был недостижим, и кандидат из
  другого города получал `STOP_PERSISTENT` вместо объяснения. Разбор — [review_20260831.md](review_20260831.md).

Ещё две правки вежливости, сделанные на tg и приезжающие с ядром (разбор —
[review_20260830.md](review_20260830.md)):

- **вводная перед доп-вопросами** (`questions_intro_sent`): текст приезжает слотом `intro` вплотную к
  первому доп-вопросу, флаг ведёт КОД;
- **эскалация переспросов и благодарность**: формулировка зависит от НОМЕРА повтора (объясни →
  предупреди), за принятый ответ ставится слот `ack`. У `field_work` своя строка предупреждения.

---

## 8. Тесты

**Их pytest — гейт, и он сломается специально.** `test_screening_engine.py` (367 строк) ассертит ровно
ту механику, которую порт отменяет: `TestCounterCaps`, `TestReaskCaps`, перерешивание хода. Он
**переписывается целиком** под ядро: вместо «движок подменил решение Аналитика» проверяется
«`decide()` вернул такой `TurnPlan`».

Что стоит покрыть (в харнессе это офлайн-гейт из 27 проверок, `runners/screening_split.py::_policy_hh_selfcheck`):

- мультиформат: отказ от одного формата → вопрос про следующий; отказ от всех → `KO_FORMAT`;
- `REMOTE` среди допустимых → `format_check: n/a`, вопроса про присутствие нет;
- `field_work`: своя ветка, свой счётчик переспросов;
- выбор `KO_LOCATION` / `KO_FORMAT` / `KO_LOCATION_GEO`;
- смена формата в вопросе не жжёт reask-бюджет;
- пороги событий (gibberish/bot_check 2, demand/pause 3), reask-cap 3, `no_progress` 4;
- зарплата: годность claim, пересчёт, вилка в тенге, отсев по нижней границе;
- миграция старой строки: вилка разобрана ДО вырезания, дефолты добиты, `last_asking` обнулён.

**Анти-луп кейсов в hh нет вовсе** — завести аналог `tests/fixtures/screening_split/counter_loops.yaml`
(настойчивый переспрашиватель обязан завершиться в пределах капа; кооперативный → `FINISH` без
ложного капа).

Гонять — из `eggplant-api`, их окружением (Python 3.14 + Postgres), не из штаба.

---

## 9. Как проверять промпты и ядро из штаба (`ai-agents`)

Всё бесплатно, кроме последней команды.

```bash
# офлайн-гейт канальной дельты ядра (27 проверок) + плумбинг
wsl.exe -e bash -lc 'cd /mnt/c/Users/user/Desktop/ANCOR/ai-agents && .venv/bin/python -m qa_harness.runners.screening_split --channel hh --offline --sample 1'

# тест нормальных диалогов, hh: 5 траекторий, ядро на заглушках
wsl.exe -e bash -lc 'cd /mnt/c/Users/user/Desktop/ANCOR/ai-agents && .venv/bin/python -m qa_harness.runners.screening_dialogue --channel hh --offline'

# ЖИВОЙ прогон hh (жжёт токены; промпт untracked, поэтому только через --prompts-path)
wsl.exe -e bash -lc 'cd /mnt/c/Users/user/Desktop/ANCOR/ai-agents && source .venv/bin/activate && set -a && source .env && set +a && python -m qa_harness.runners.screening_dialogue --channel hh --prompts-path ../prompts'
```

Первые две печатают `[offline] hh-ядро policy … 27/27` и `[dialogue] passed=4/4 · errors=0 ·
skipped=1`. Третья — пять разборов диалогов и `passed=5/5`; порядок цены — сотни тысяч токенов.
**Офлайн качество промпта НЕ проверяет:** там скриптуется само НАБЛЮДЕНИЕ, настоящие только `decide()`
и счётчики.

---

## 10. Открытые решения и порядок

1. **Комментарии и докстринги** в файлах порта: в tg-PR их сняли по требованию владельца репо, для
   eggplant решать отдельно.
2. **Куда писать аудит зарплатного claim** (в tg — коллекция Mongo).
3. **Б3, кандидаты за границей.** В hh остался рудимент: `KO_LOCATION_GEO` отсеивает только при
   двойном совпадении (кандидат явно сказал, что за рубежом, И в вакансии явно прописано
   гео-ограничение). Работает случайно, а не по правилу. В hh-ядро перенесено КАК ЕСТЬ; порт — не
   место для новой политики, но и тихо потерять отсев нельзя.

**Порядок работ:**

1. ядро в `eggplant-api` — `policy/` + зарплатный модуль + async-обвязка; гарды внутри движка
   (отдельной точки санитайзера в `HHClient.send_message`, `app/hh/client.py:129`, заводить не надо —
   шлюз стоит в движке);
2. хранилище — Alembic-ревизия под `salary_band` + чтение `currency` из вакансии (§5.1), ленивая
   миграция старых строк с громким бэкфиллом (§5.2);
3. `REASK_CAP` → 3 по новой семантике;
4. тесты — переписать `test_screening_engine.py`, завести анти-луп кейсы;
5. промпт: закоммитить `screening_analyzer_hh/v3` в `prompts` и выпустить релиз — **после** того как
   код лёг в master.

**Приёмка:** их pytest зелёный · офлайн-гейт харнесса 27/27 и `screening_dialogue --channel hh
--offline` 4/4 не сломаны · живой прогон hh-канала хотя бы раз сделан и разобран · пороги каналов
сошлись (reask 3, события 2/3, `no_progress` 4) · вилка в тенге не сравнивается как рублёвая.

---

## 11. Чего в новом ядре нет и быть не может

Чтобы не искать в коде то, чего там нет. Все девять конструкций старого движка, компенсировавших
порядок «модель решила раньше, чем код посчитал», **невыразимы** в новом контракте: `_gate_salary_update`,
`_assumes_salary_closed`, `_release_money_stop`, `_SALARY_REWIND_NOTE`, флаг `_forced`, отложенный
инкремент `event`, три ветки перерешивания хода. Поля `next_action`, `script_key`, `instruction`,
`asking`, `event` в контракте `Observation` отсутствуют — исход модель вернуть не может, его негде
выразить.

Что осталось за моделью и порт этого не лечит: распознавание сигналов (код может не исполнить
услышанное, но не может услышать за неё), заполнение `salary_claim` кроме `scale`/`currency`
(прикрыты гейтами), `focus_answered`, и всё, что делает Интервьюер.

# Статус миграции раннеров на новую архитектуру (`qa_harness`)

Живой чек-лист: что переносим из каждого старого `app/`-раннера и что уже сделано
(чтобы ничего не забыть). Старые раннеры НЕ удаляем до проверки новых и финального cutover.

> Решение по парити (2026-06-02): **автоматический парити-харнесс на кассетах НЕ строим** —
> эквивалентность нового раннера старому проверяется **вручную, глазами** по отчётам.
> Колонка «Парити» = статус ручной сверки (⬜ не сверяли / 👁 сверено глазами).

Легенда: ✅ перенесено · 🟡 частично · ⬜ не начато · ➕ новое (нет в старом)

## Сводка по раннерам

| Раннер | Статус | Парити | Старый файл |
|---|---|---|---|
| message_classifier | ✅ фичи | ⬜ глазами | `app/message_classifier_runner.py` |
| verdict_classifier | ✅ фичи | ⬜ глазами | `app/verdict_classifier_runner.py` |
| extractor_agent | ✅ фичи | 👁 глазами | `app/extractor_agent_runner.py` |
| sourcing_assistant | ✅ фичи | 👁 глазами | `app/sourcing_assistant_runner.py` |
| one_line_search_query_builder | ✅ фичи | 👁 глазами | `app/one_line_search_query_builder_runner.py` |
| responsibilities_parser | ✅ фичи | 👁 глазами | `app/responsibilities_parser_runner.py` |
| screening_autofill | ✅ фичи | 👁 глазами | `app/screening_autofill_runner.py` |
| screening_guardrails | ✅ фичи | 👁 глазами | `app/screening_guardrails_runner.py` |
| screening_scenarios (std) | ✅ фичи (CSV+LLM-судья) | 👁 глазами (7/7) | `app/screening_scenarios_runner.py` |
| screening_scenarios_hh | ✅ фичи (`--component`) | — (нечего гонять: 0 примеров) | `app/screening_scenarios_hh_runner.py` |
| first_touch (base) | ✅ фичи | 👁 глазами | `app/first_touch_runner.py` |
| first_touch_hh | ✅ фичи | 👁 глазами | `app/first_touch_hh_runner.py` |
| first_touch_event | ✅ фичи | 👁 глазами | `app/first_touch_event_runner.py` |
| verdict/message: общие фичи | — | — | — |

Порядок (от непохожих к похожим, чтобы трудные формы всплыли рано):
message_classifier → verdict_classifier → **extractor** (stages[]) → sourcing (subjects[]) →
one_line/responsibilities → first_touch → screening (std→hh).

---

## message_classifier — детальный чек-лист

Старый раннер: `python -m app.message_classifier_runner` (по умолчанию `--mode synthetic`).
Новый раннер: `python -m qa_harness.runners.message_classifier`.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| 4 класса (`reason_farewell/no_reason/acceptance/human_needed`) | ✅ | `domain/judge/label_judge.py` (`CLASSES`) |
| `_extract_label` (regex по меткам) | ✅ | `domain/judge/label_judge.py` (`extract_label`) |
| Классификация stored-промптом (онлайн) | ✅ | `domain/classifiers/message.py` (`StoredPromptMessageClassifier`) |
| Метрики: accuracy, per-class, confusion matrix | ✅ | `core/metrics.py` |
| Regression-кейсы (`--mode regression`) | ✅ | `runners/message_classifier.py` |
| Two-file отчёт + учёт токенов | ✅ | `core/reporting.py` |
| Офлайн-классификация без сети | ➕ | `HeuristicMessageClassifier` (новое) |
| **Синтетическая генерация** (`--messages-per-class`) | ✅ | `domain/generators/` |
| `CandidateMessageSynthesizer` (LLM из CDM-контекста) | ✅ | `domain/generators/message_gen.py` (`CandidateMessageGenerator`) |
| `_validate_generated_message` (валидация соответствия классу) | ✅ | `domain/generators/message_gen.py` (`validate_candidate_message`) |
| Сценарные подсказки `SCENARIO_HINTS_BY_CLASS` | ✅ | `domain/generators/message_specs.py` |
| `--scenario-mode` (random/cycle), `--scenario-count-per-class` | ✅ | runner |
| `--noise-level` (0..2) | ✅ | runner |
| `--max-attempts-multiplier` (ретраи генерации) | ✅ | runner |
| `--mode synthetic\|regression\|all` | ✅ | runner |
| `--cdm-dir / --cdm-count`, `--message-gen-model` | ✅ | runner |
| Сплиты в отчёте: synthetic vs regression | ✅ | `metrics.classification.by_split` |
| Раздельный учёт токенов generator/classifier | ✅ | `metrics.token_usage_by_role` |
| Парити-сверка со старым раннером | ⬜ | (следующий шаг) |

Отличие по умолчанию: старый раннер по умолчанию `--mode synthetic` (и требует
`--messages-per-class`); новый по умолчанию `--mode regression` (запуск без флагов = 8 кейсов,
без жжения токенов). Полная синтетика доступна явно: `--mode all --messages-per-class N`.

---

## verdict_classifier — детальный чек-лист

Старый: `python -m app.verdict_classifier_runner`. Новый: `python -m qa_harness.runners.verdict_classifier`.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Вердикты `passed/failed/deadlock` + `_extract_verdict` | ✅ | переиспользует `extract_label(text, VERDICTS)` + `LabelJudge` |
| Классификация диалога stored-промптом (онлайн) | ✅ | `domain/classifiers/verdict.py` (`StoredPromptVerdictClassifier`) |
| Офлайн-классификация без сети | ➕ | `HeuristicVerdictClassifier` (стаб) |
| Регрессионные кейсы (`--mode regression`) | ✅ | `runners/verdict_classifier.py` |
| `DialogueSynthesizer` (генерация диалога под вердикт) | ✅ | `domain/generators/dialogue_gen.py` (`DialogueGenerator`) |
| Валидация диалога (формат/чередование/END/утечка зарплаты/deadlock) | ✅ | `validate_generated_dialogue` |
| Сценарные подсказки `SCENARIO_HINTS_BY_VERDICT` + marker-константы | ✅ | `domain/generators/dialogue_specs.py` |
| Хелперы формата диалога (split lines, speaker, prefixes) | ✅ | `domain/text/dialogue.py` (общие, для screening тоже) |
| `--mode/--dialogs-per-verdict/--noise-level/--scenario-*/--max-attempts-multiplier/--dialogue-gen-model` | ✅ | runner |
| Метрики + by_split (synthetic vs regression) + токены gen/clf | ✅ | `core/metrics` + `metrics.token_usage_by_role` |
| Структурированный транскрипт диалога в cases.json | ➕ | `dialogue_to_transcript` (новое; старый писал плоский текст) |
| Парити-сверка со старым раннером | ⬜ | (следующий шаг) |

Переиспользование каркаса (доказательство обобщаемости): судья `LabelJudge`, метрики,
`ReportBuilder`, генератор-база `Generator`, `core/cdm` — взяты из message_classifier без изменений;
новое — только диалоговая специфика (`dialogue_gen`/`dialogue_specs`/`verdict`-классификатор).

---

## extractor_agent — детальный чек-лист

Старый: `python app/extractor_agent_runner.py`. Новый: `python -m qa_harness.runners.extractor_agent`.
Структурно ДРУГОЙ: конвейер step1→2→3, без LLM-судьи, оценка контрактная; кейс в отчёте = `stages[]`.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Step1: LLM-парс запроса -> extractor_json (сырой HTTP /responses) | ✅ | `pipeline/openai_step1.py` |
| Step1 contract-валидация (ALLOWED_*, drift) | ✅ | `pipeline/contract.py` |
| Step2: extractor_json -> backend payload (группы/флаги/диапазоны/гео-санитайз) | ✅ | `pipeline/payload.py` (`build_step3_payload`) |
| Step3: backend `/site/searchBool` + классификация ошибок (insufficient/auth/http) | ✅ | `pipeline/backend_client.py` |
| Источники кейсов real / suite / synthetic | ✅ | `pipeline/cases.py` |
| `--steps 1\|1,2\|1,2,3` (частичный прогон) | ✅ | runner |
| backend-флаги (base-url/token/step3-path/timeout/retries/token-in-body) + search-флаги | ✅ | runner |
| `stages[]` в отчёте (артефакт + pass/fail на каждый шаг) | ➕ | `core/reporting` (subjects/stages) — впервые использовано |
| Метрики: deterministic (contract/ошибки) + backend (counts) + by_source | ✅ | `metrics.deterministic/backend/by_source` |
| `--mix-ratios` (сэмплинг real/suite/syn по долям) | 🟡 | `pipeline/cases.mix_cases` есть и протестирован; раннер v1 пока простой concat по счётчикам |
| Парити-сверка со старым | ⬜ глазами | (вручную по отчётам) |

`stages[]` отработал на конвейерной форме — схема отчёта (спроектированная против classifier+extractor, P0-5)
держит и многошаговый раннер без переписывания. `pipeline/` — общий слой для будущих one_line/sourcing.

**Пересборка v2 (см. [EXTRACTOR_REDESIGN.md](EXTRACTOR_REDESIGN.md)):** кейсы → курируемые якоря с golden
(`anchors.yaml`), real(263)/suite/synthetic удалены; добавлена **семантическая** оценка (`domain/extractor/semantic.py`)
поверх контракта; **поэтапные вердикты**; **качество промпта ≠ инфраструктура** (backend-сбои → `errors`, не `failed`);
step1 на SDK (`StoredPromptClient`) + строгий парс (ok/dirty/invalid); step2 coverage (`mapping_report`);
конкурентность + раздельные таймауты + fail-fast по бэкенду + чекпоинты/сохранение по Ctrl+C.

**Общий цикл прогона вынесен в `core/run_loop.py`** (`run_cases`, [тесты](../tests/test_run_loop.py)):
конкурентный fan-out + последовательный fold в главном потоке (без локов) + чекпоинты + сохранение по Ctrl+C.
extractor_agent и one_line_search_query_builder — потребители; fail-fast (`backend_down`) остаётся в раннере как
shared state — цикл о нём не знает. Будущие раннеры (sourcing, …) получают оркестрацию даром (план: «вынести `core/run_loop`» — done).

---

## one_line_search_query_builder — детальный чек-лист

Старый: `python -m app.one_line_search_query_builder_runner`. Новый: `python -m qa_harness.runners.one_line_search_query_builder`.
Тестируем промпт-**билдер** (вакансия → однострочный boolean-запрос). Конвейер step1(builder)→step2(extractor)→step3(backend),
оценка поэтапная, кейс в отчёте = `stages[]` (как extractor). Переиспользует `core/run_loop` + `pipeline/`.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| step1: builder-промпт vacancy → query (сырой /responses) | ✅ | `core.StoredPromptClient` (SDK, ретраи, раздельный таймаут) |
| Формальные проверки запроса (одна строка / не пусто / без JSON) | ✅ | `domain/query_builder/checks.py` (`build_query_checks`) |
| Анти-утечка (формат работы / зарплата / процесс найма) | ✅ | `domain/query_builder/checks.py` (`detect_leakage`, бывш. `FORBIDDEN_PATTERNS`) |
| step2: query → extractor-промпт → extractor_json + payload | ✅ | `pipeline/` (`parse_extractor_json`/`validate_step1_contract`/`build_step3_payload`/`mapping_report`) |
| step3: backend `/site/searchBool` → count | ✅ | `pipeline/backend_client.py` |
| `--steps 1\|1,2\|1,2,3`, backend/search-флаги, `--token-in-body` | ✅ | runner (как extractor) |
| Two-file отчёт + `stages[]` + конкурентность/чекпоинты/Ctrl+C | ✅ | `core/reporting` + `core/run_loop` |
| Раздельный override промпта extractor (`--extractor-prompt-id/version`) | ✅ | runner (резолв `extractor_agent` из `model.yaml`) |
| Офлайн без сети | ➕ | `--offline` replay `offline_query` из golden |
| Парити-сверка со старым | ⬜ глазами | (вручную по отчётам) |

**Осознанные отличия от легаси:**
- источник кейсов: CDM-вакансии → курируемые **golden-вакансии** (`tests/fixtures/one_line_search_query_builder/golden.yaml`) с `expect`/`forbid` на строке запроса;
- мёртвый `evaluate_semantics`/anchor-coverage (в `main()` не вызывался) заменён живой **golden-семантикой** (`domain/query_builder/semantic.py`);
- **quality ≠ infra**: `passed = format & no_leakage & semantic`; backend `count>0` больше НЕ гейтит качество (это retrieval-инфо), инфра-сбои builder/extractor/backend → `errors`, не `failed`;
- по умолчанию `--steps 1` (качество билдера полностью на step1; 2/3 — downstream-инфо);
- схема `report.cases.schema.json` дополнена `source: anchor|golden` — заодно стал валиден и extractor (он эмитит `source=anchor`).

---

## sourcing_assistant — детальный чек-лист

Старый: `python -m app.sourcing_assistant_runner`. Новый: `python -m qa_harness.runners.sourcing_assistant`.
Тестируем промпт sourcing_assistant (кандидат ↔ требования вакансии). Кейс = **ВАКАНСИЯ**, `subjects[]` = N
кандидатов. Переиспользует `core/run_loop` + `pipeline/` (backend-поиск). **Первый раннер с `subjects[]`.**

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Требования 1..5 из CDM (key_requirements / stack+skills) | ✅ | `domain/sourcing/build.py` (`requirements_from_cdm`) |
| backend-поиск реальных кандидатов по `extractor_entities` (limit=pool) | ✅ | runner + `pipeline` (`build_step3_payload`/`call_backend_search_bool`) |
| Сэмпл N профилей (first/random) | ✅ | runner (`--candidate-sample-size`/`--sample-mode`) |
| Профиль кандидата из backend-выдачи (about/skills/positions) | ✅ | `domain/sourcing/build.py` (`build_candidate_profile`) |
| Промпт на каждого кандидата `{requirements, profile}` → массив | ✅ | runner + `core.StoredPromptClient` |
| Контракт вывода (длина 1:1, форма {requirement,comment,passed}, точный echo) | ✅ | `domain/sourcing/contract.py` (`check_contract`/`parse_sourcing_output`) |
| Two-file отчёт + `subjects[]` + конкурентность/чекпоинты/Ctrl+C | ✅ | `core/reporting` (subjects[]) + `core/run_loop` |
| Источник требований `responsibilities_parser` (LLM) | ⬜ | пропущен (есть `cdm_key_requirements`/`stack_skills`; это отдельный раннер) |
| Офлайн без сети | ➕ | `--offline` replay канонных кандидатов (`tests/fixtures/sourcing_assistant/offline.yaml`) |
| Парити-сверка со старым | ⬜ глазами | (вручную по отчётам) |

**Осознанные отличия от легаси:**
- `case.passed` = ВСЕ оценённые кандидаты прошли контракт (легаси: «≥1 прошёл» — мягче; новое строже и честнее для теста надёжности формата);
- **quality ≠ infra**: backend-сбои / нет entities / нет кандидатов / сетевые ошибки промпта → `errors`, не `failed`; в `cases[]` попадают только «оценённые» вакансии;
- семантической оценки нет (как и в легаси): кандидаты живые, без разметки — только контракт формы;
- дефолты мелкие (`--candidate-pool-size 10`, `--candidate-sample-size 5`): backend-профили (`limit>0`) — медленный путь.

---

## responsibilities_parser — детальный чек-лист

Старый: `python -m app.responsibilities_parser_runner`. Новый: `python -m qa_harness.runners.responsibilities_parser`.
**Без бэкенда** (только LLM): текст вакансии → JSON-массив ключевых терминов. Структурно как one_line (golden).

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Промпт vacancy → JSON-массив строк (ключевые слова) | ✅ | `core.StoredPromptClient` |
| Контракт: 1..5 терминов, 1..3 слова, без чисел-одиночек/запятых, ≤60 | ✅ | `domain/responsibilities/contract.py` |
| Без дублей (нормализованных) | ✅ | `contract.find_duplicates` |
| Заземление: термины найдены в тексте вакансии | ✅ сигнал | `domain/responsibilities/semantic.py` (`grounding_misses`) |
| Совпадение с ожидаемым (`vacancy_stack∪skills`) | заменено | golden `expect`/`forbid` (`check_semantics`) |
| Two-file отчёт + конкурентность/чекпоинты | ✅ | `core/reporting` + `core/run_loop` |
| Офлайн без сети | ➕ | `--offline` replay `offline_output` из golden |
| Парити-сверка | ⬜ глазами | (вручную) |

**Осознанные отличия от легаси:**
- источник кейсов: CDM-вакансии → курируемые **golden-вакансии** (`tests/fixtures/responsibilities_parser/golden.yaml`);
- `passed = contract & semantic(golden)`; заземление в тексте — **сигнал-предупреждение** (не gate), как warning в легаси;
- семантика по golden `expect`/`forbid` вместо нечёткого матчинга против `vacancy_stack∪skills`;
- grounding — строгая подстрока (lower+ё→е, без стемминга) → может ложно метить склонённые формы (напр. «микросервисы» vs «микросервисов»); это лишь сигнал, качество не валит;
- quality ≠ infra: сетевой сбой промпта → `errors`; невалидный JSON-вывод → contract-фейл (качество).

---

## screening_autofill — детальный чек-лист

Старый: `python -m app.screening_autofill_runner`. Новый: `python -m qa_harness.runners.screening_autofill`.
**Без бэкенда** (только LLM): диалог → JSON-форма скрининга. Объём — **curated-golden** (решение пользователя):
синтетик-LLM-генератор диалогов и детерминированные билдеры (~600 строк) НЕ переносим, берём ручные диалоги.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Промпт диалог → форма {location, salary×2, work_format, additional_info} | ✅ | `core.StoredPromptClient` (обёртка-инструкция как в легаси-клиенте) |
| Схема формы (ключи/типы/enum work_format/digits зарплат/форма additional_info) | ✅ | `domain/screening_autofill/contract.py` (`validate_schema`) |
| Расплющивание диалога в одну строку (как прод) | ✅ | runner (`_flatten`; `--no-flatten` чтобы отключить) |
| 12 work_format-сценариев (explicit/silent/ignored/rejected/multiple/…) | 🟡 | свёрнуты в golden-диалоги (hybrid/remote/office/silent/recruiter-only/rejects) |
| Извлечение зарплата/локация/формат | ✅ | golden `expect` (work_format точно; зарплата/локация — `<nonempty>`) |
| Анти-утечка в additional_info (темы + метки спикера) | ✅ | `domain/screening_autofill/semantic.py` (`additional_info_leaks`) |
| additional_info непустой при не-исключённом вопросе | ✅ | `expect_additional_info_nonempty` в golden |
| Two-file отчёт + конкурентность/чекпоинты | ✅ | `core/reporting` + `core/run_loop` |
| Офлайн без сети | ➕ | `--offline` replay `offline_output` из golden |
| Синтетик-генератор диалогов (LLM) + детерминир. билдеры | ⬜ | НЕ переносим (curated-golden); при нужде — отдельно |
| Парити-сверка | ⬜ глазами | (вручную) |

**Осознанные отличия от легаси:**
- источник кейсов: синтетика + регрессия-билдеры → курируемые **golden-диалоги** (`tests/fixtures/screening_autofill/golden.yaml`, 10 шт);
- `passed = schema & expect(golden) & no_leak & (additional_info_nonempty?)`; в легаси semantic для регрессии был warning — здесь no_leak/expect это gate;
- зарплата/локация — `<nonempty>` (точный формат варьируется), work_format — точным значением (enum стабилен);
- quality ≠ infra: сетевой сбой → `errors`; невалидный JSON → schema-фейл (качество).

---

## first_touch (base) — детальный чек-лист

Старый: `python -m app.first_touch_runner`. Новый: `python -m qa_harness.runners.first_touch`.
**Без бэкенда**; генерация первого касания → **LLM-судья фактов** + эвристики. Объём — **база + LLM-судья**
(решение пользователя): `_hh`/`_event` и нишевые prompt_rule/possessive-проверки пока НЕ переносим.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Генерация first_touch (payload → сообщение) | ✅ | `core.StoredPromptClient` (text-формат + срез подписи) |
| LLM-судья: facts_present / hallucinated_facts / question | ✅ | `domain/first_touch/judge.py` (`FactJudge` на `ModelClient` — первый LLM-судья) |
| extra_numbers (выдуманные числа) | ✅ ≥5 цифр | `domain/first_touch/checks.py` |
| company_hidden — нет утечки названия | ✅ | `checks.company_name_leaked` |
| Two-file отчёт + конкурентность/чекпоинты | ✅ | `core/reporting` + `core/run_loop` |
| Офлайн без сети | ➕ | `--offline` replay offline_message + эвристика вместо судьи |
| Варианты `_hh` / `_event` | ⬜ | пока не переносим (отдельно) |
| Нишевые prompt_rule / possessive-source проверки | ⬜ | не переносим (curated-golden) |
| Парити-сверка | ⬜ глазами | (вручную) |

**Осознанные отличия от легаси:**
- источник кейсов: CDM + possessive + prompt_rule → курируемые **golden** (`tests/fixtures/first_touch/golden.yaml`, 6 шт: видимая/скрытая компания, с зарплатой/без, formal/informal);
- `passed = facts(required) & no_extra_numbers & question & company_hidden`; **галлюцинации — сигнал, не gate** (LLM-судья шумит на общих фразах; в легаси тоже strict-only); `company_description`/`responsibilities` авто-добавляются в allowed-факты судьи; при `company_hidden` название НЕ передаётся генератору;
- extra_numbers гейтит только числа ≥5 цифр (зарплата-величина), чтобы не ловить ложно годы/счётчики;
- зарплата — `optional_facts` (отсутствие не валит passed);
- `--offline` использует эвристику фактов (стем-токены) вместо LLM-судьи; галлюцинации офлайн не ловятся;
- quality ≠ infra: сбой генерации/судьи → `errors`.

**Вариант `first_touch_hh`** (`runners/first_touch_hh.py`, тонкая обёртка): тот же конвейер через
`--component first_touch_hh` + своя golden-фикстура + правило `forbid_in_message` (HH — источник НЕ упоминать,
проверка `forbidden_phrases`). База `first_touch` обобщена: payload = `case.input`, `--component` выбирает промпт.
**`first_touch_event`** (`runners/first_touch_event.py` + `domain/first_touch_event/`) — фиксированное
мероприятие VK JT Go, свой **EventJudge** (missing/hallucinated/forbidden_claims против эталона) + эвристики
(greeting «Имя, здравствуйте!», финальный вопрос про регистрацию, extra_numbers с allow={4}). golden = имена
кандидатов + offline_message; payload генерации = `{candidate_name}`. `passed = greeting & final_question &
no_missing & no_hallucinated & no_forbidden & no_extra_numbers`.

---

## screening_guardrails — детальный чек-лист

Старый: `python -m app.screening_guardrails_runner`. Новый: `python -m qa_harness.runners.screening_guardrails`.
Гоняет **ЖИВОЙ** промпт `screening_assistant` в мультитёрн-разговоре и ловит нарушения-гардрейлы в каждом
ответе. Новая общая инфра разговора — `domain/screening/conversation.py` (Conversations API, БЕЗ легаси
`screeningAssistant/`), переиспользуется и для scenarios.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Мультитёрн со screening_assistant (Conversations API + сид вакансии) | ✅ | `domain/screening/conversation.py` (`ScreeningConversation`) |
| LLM-судья 3 нарушений (self_answer / repeated_questions / premature_end) | ✅ | `domain/screening_guardrails/judge.py` (`GuardrailJudge`) |
| Эвристики-фолбэк + жёсткий гейт (нет вопросов → premature=false) | ✅ | `domain/screening_guardrails/detectors.py` |
| Per-turn разметка флагов в отчёте | ✅ | `transcript[].flags` (схема) |
| Two-file отчёт + конкурентность/чекпоинты | ✅ | `core/reporting` + `core/run_loop` |
| Офлайн без сети | ➕ | `--offline` replay `offline_turns` + эвристики (без судьи) |
| Синтетик-генератор диалогов кандидата (GEN_MODEL) | ⬜ | заменён курируемыми golden-разговорами |
| Парити-сверка | ⬜ глазами | (вручную) |

**Осознанные отличия от легаси:**
- источник разговоров: синтетик-генерация → курируемые **golden** (`tests/fixtures/screening_guardrails/golden.yaml`);
- `passed` (кейс = разговор) = ни один ответ ассистента не нарушил гардрейлы;
- судья при сбое/непарсе graceful-фолбэк на эвристики (как легаси), видно по `verdict.meta.used_heuristics`;
- `--offline` гоняет эвристики на canned `offline_turns` (без живого ассистента/судьи);
- quality ≠ infra: сбой разговора (Conversations API) → `errors`, не `failed`.

## screening_scenarios (+hh) — детальный чек-лист

Старый: `python -m app.screening_scenarios_runner` / `..._hh_runner` (~4000 строк hardcoded-эвристик и
chain-групп на сценарий). Новый: `python -m qa_harness.runners.screening_scenarios` (`--component
screening_assistant` | `screening_assistant_hh`). Сценарии берём из того же CSV-golden
(`tests/fixtures/screening_scenarios.csv`, hh — `screening_scenarios_hh.csv`): из примеров диалогов
вытаскиваем реплики кандидата, гоняем живой screening_assistant (общая `domain/screening/conversation.py`),
а `ScenarioJudge` (LLM) судит транскрипт против `expected_behavior` сценария.

| Фича старого раннера | Статус | Где в новом |
|---|---|---|
| Сценарии из CSV (название/описание/ожидание/примеры) | ✅ | `domain/screening_scenarios/cases.py` (`load_scenarios`) |
| Извлечение реплик кандидата из примеров (инлайн `[candidate]`) | ✅ | `cases.py` (`extract_candidate_examples`, raw_decode JSON-объектов) |
| Мультитёрн со screening_assistant | ✅ | `domain/screening/conversation.py` (общая с guardrails) |
| Оценка соответствия ожидаемому поведению | ✅ (LLM-судья) | `domain/screening_scenarios/judge.py` (`ScenarioJudge`) |
| hh-вариант (промпт `screening_assistant_hh`) | ✅ | `--component screening_assistant_hh` (+свой CSV) |
| `--scenario-indices` / выборка | ✅ | `--scenario-indices` / `--sample N` (0 = все с примерами) |
| Two-file отчёт + конкурентность/чекпоинты | ✅ | `core/reporting` + `core/run_loop` |
| Офлайн-плумбинг (load+extract без сети) | ➕ | `--offline` |
| ~4000 строк hardcoded-эвристик и chain-групп | ⬜ | заменены LLM-судьёй против `expected_behavior` |
| Парити-сверка | 👁 глазами | base 7/7 passed (2026-06-12), 0 infra-ошибок; hh — нечего гонять (0 примеров) |

**Осознанные отличия от легаси:**
- hardcoded-эвристики/chain-группы → один LLM-судья (`ScenarioJudge`) против `expected_behavior` из CSV;
- онлайн гоняются ТОЛЬКО сценарии с примерами диалога кандидата (base CSV: **7/62**, hh CSV: **0/50** —
  у остальных в CSV пусто поле примеров); сценарий без примеров → **skip** (`metrics.scenarios.skipped_no_examples`), не `failed`/`errors`;
- `passed` (кейс = сценарий) = поведение ассистента соответствует `expected_behavior` по сути (тон/формулировки не важны);
- quality ≠ infra: сбой разговора/судьи → `errors`, не `failed`.

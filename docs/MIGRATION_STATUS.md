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
| extractor_agent | ✅ фичи | ⬜ глазами | `app/extractor_agent_runner.py` |
| sourcing_assistant | ⬜ | ⬜ | `app/sourcing_assistant_runner.py` |
| one_line_search_query_builder | ⬜ | ⬜ | `app/one_line_search_query_builder_runner.py` |
| responsibilities_parser | ⬜ | ⬜ | `app/responsibilities_parser_runner.py` |
| screening_autofill | ⬜ | ⬜ | `app/screening_autofill_runner.py` |
| screening_guardrails | ⬜ | ⬜ | `app/screening_guardrails_runner.py` |
| screening_scenarios (std) | ⬜ | ⬜ | `app/screening_scenarios_runner.py` |
| screening_scenarios_hh | ⬜ | ⬜ | `app/screening_scenarios_hh_runner.py` |
| first_touch / _hh / _event | ⬜ | ⬜ | `app/first_touch*_runner.py` |
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
extractor_agent — первый потребитель; fail-fast (`backend_down`) остался в раннере как его shared state — цикл
о нём не знает. Будущие sourcing/one_line получат оркестрацию даром (см. план: «вынести `core/run_loop`» — done).

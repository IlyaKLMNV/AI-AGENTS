# Статус миграции раннеров на новую архитектуру (`qa_harness`)

Живой чек-лист: что переносим из каждого старого `app/`-раннера и что уже сделано.
Парити-сверка ловит поведенческие расхождения; этот файл ловит **что вообще портируем**
(чтобы ничего не забыть). Старые раннеры НЕ удаляем до проверки новых и финального cutover.

Легенда: ✅ перенесено · 🟡 частично · ⬜ не начато · ➕ новое (нет в старом)

## Сводка по раннерам

| Раннер | Статус | Парити | Старый файл |
|---|---|---|---|
| message_classifier | ✅ фичи (парити ⬜) | ⬜ | `app/message_classifier_runner.py` |
| verdict_classifier | ✅ фичи (парити ⬜) | ⬜ | `app/verdict_classifier_runner.py` |
| extractor_agent | ⬜ | ⬜ | `app/extractor_agent_runner.py` |
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

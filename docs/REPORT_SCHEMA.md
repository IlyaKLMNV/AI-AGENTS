# Универсальная схема отчётов QA-харнесса

> Применима ко всем ~12 промптам. Три файла на прогон (два JSON + человекочитаемый `*.review.md`). JSON-схемы: см.
> [schemas/report.metrics.schema.json](schemas/report.metrics.schema.json) и
> [schemas/report.cases.schema.json](schemas/report.cases.schema.json).

## 1. Зачем разделение на файлы

Сейчас каждый раннер пишет один монолитный JSON, где 65–86% объёма — текст диалогов,
продублированный ещё раз в `mismatches`. Guardrails уже вынужденно завёл `*_compact.json`
(выигрыш ~83×) — это эмпирическое доказательство, что разделение нужно.

| Файл | Содержит | Размер | Для кого |
|---|---|---|---|
| `<runner>_<run_id>.metrics.json` | `meta` + `summary` + `metrics` + `failures_index` | КБ | дашборды, тренды по версиям промпта, диффы, CI-gate |
| `<runner>_<run_id>.cases.json` | `cases[]`: входы, транскрипт, выход, вердикты | КБ–МБ | drill-down при разборе падений, реплей |
| `<runner>_<run_id>.review.md` | человекочитаемый рендер из тех двух (по кейсу: ожидание → диалог → вердикт; провалы первыми, прошедшие свёрнуты) | КБ–МБ | разбор глазами |

Связь: одинаковый `meta.run_id` + `schema_version` в обоих; `failures_index[].case_id`
ссылается на `cases[].case_id`. Сырой текст хранится **ровно один раз** — в `cases.json`.
`review.md` — производная человекочитаемая проекция тех же данных (рендерится из обоих документов
в `core/reporting.render_review_md`); не источник правды, перегенерируется в любой момент.

Все три файла пишутся строго **UTF-8 без BOM** (чинит текущий разнобой `utf-8-sig`/BOM в 4 раннерах).

## 2. Файл МЕТРИК — стабильная оболочка у всех раннеров

```jsonc
{
  "schema_version": "1.0",
  "kind": "metrics",
  "meta": {
    "run_id": "20260601_142233",
    "started_at": "2026-06-01T14:22:33Z", "finished_at": "...", "duration_s": 148.2,
    "runner": "screening_scenarios",                 // фиксированный enum
    "prompt_under_test": { "component": "screening_assistant", "prompt_id": "pmpt_…", "prompt_version": "49" },
    "models": { "generator": "gpt-4.1-mini", "evaluator": "gpt-4.1" },  // null где роли нет
    "seed": 1234, "git_commit": "5584f96",
    "args": { "csv_path": "…", "mode": "scenario+chain" }
  },
  "summary": {                                        // ОДИНАКОВО У ВСЕХ
    "total": 63, "passed": 61, "failed": 2, "errors": 0, "pass_rate": 96.83,
    "token_usage": { "input": 185327, "output": 25531, "total": 210858 }
  },
  "metrics": { /* расширяемо; предсказуемые под-ключи ниже */ },
  "failures_index": [
    { "case_id": "regression:wf_hybrid:v1", "reason_codes": ["location_missing"], "severity": "high", "one_line": "…" }
  ]
}
```

`meta.runner` (enum): `screening_scenarios`, `screening_scenarios_hh`, `screening_guardrails`,
`message_classifier`, `verdict_classifier`, `screening_autofill`, `first_touch`, `first_touch_hh`,
`first_touch_event`, `extractor_agent`, `one_line_search_query_builder`, `responsibilities_parser`,
`sourcing_assistant`.

Под-ключи `metrics.*` (раннер заполняет только применимые):

| Ключ | Когда | Содержимое |
|---|---|---|
| `judge` | LLM-судья (scenarios, hh, first_touch, guardrails) | распределение баллов, гистограмма reason-кодов |
| `classification` | классификаторы | `accuracy`, `per_class_accuracy`, разреженная `confusion_matrix`, `counts_*`, срезы synthetic/regression |
| `deterministic` | детерминированные проверки | частота срабатывания каждого правила/issue-кода |
| `backend` | поиск (extractor, one_line, sourcing) | `step3_calls`, `retrieval_ok`, `insufficient`, `avg_candidates` |
| `by_source` | где есть cdm/regression/synthetic | разбивка `total/passed` по источнику |

`summary` и `failures_index` **вычисляются `ReportBuilder` из вердиктов**, а не пишутся руками
(убирает рассинхрон вроде `accuracy`/`overall_accuracy`). `metrics.*.reason_codes` — гистограмма
тех же кодов по прогону → виден доминирующий режим отказа.

## 3. Файл КЕЙСОВ — один массив, транскрипт один раз

```jsonc
{
  "schema_version": "1.0", "kind": "cases",
  "run_id": "20260521_163326", "runner": "screening_scenarios",
  "cases": [
    {
      "case_id": "single:S6:v1",
      "source": "synthetic",                 // cdm | regression | synthetic | suite | real
      "passed": false,
      "inputs": {
        "criterion": "Не отвечать за кандидата; задать вопрос по вакансии",   // ОБЯЗАТЕЛЬНО (см. §5)
        "scenario": { "name": "S6", "index": 6 },
        "vacancy_ref": { "title": "QA Automation Lead", "company": "Indigosoft", "work_format": "hybrid", "location": "Москва" },
        "cdm_file": "tests/fixtures/cdm/std/cdm_06.json"
      },
      "transcript": [                         // полный диалог; кумулятивный dialog_text НЕ дублируем
        { "turn": 1, "role": "candidate", "text": "…" },
        { "turn": 1, "role": "assistant",  "text": "…" }
      ],
      "output": { "raw": "…", "parsed": null },   // parsed — для классификаторов/autofill
      "verdict": {                            // ОДИН обязательный вердикт (см. §4)
        "evaluator": "llm_judge", "model": "gpt-4.1", "passed": false, "score": 0,
        "reason_codes": ["self_answer"], "comment": "…", "turn_ref": 2
      },
      "checks": [                             // ОПЦИОНАЛЬНЫЙ детерминированный слой поверх
        { "rule": "has_end_marker", "passed": true },
        { "rule": "repeated_question", "passed": true }
      ]
    }
  ]
}
```

### Опциональные под-уровни (поправка критика: схема течёт на 2 из 4 семейств)
- **`subjects[]`** — для `sourcing` (кейс = вакансия, внутри N кандидатов; `case.passed` = «**все**
  оценённые кандидаты прошли контракт», строже легаси «≥1»). Каждый subject:
  `{id, passed, candidate_data, requirement_results[], verdict}`, где `candidate_data` — вход промпта
  (что увидел о кандидате), `requirement_results[].passed` — СЫРОЙ вывод промпта `0/1`.
- **`stages[]`** — для `extractor`/`one_line` (конвейер step1→step2→step3 с тремя артефактами и
  тремя точками отказа). Каждый stage: `{name, artifact, passed, reason_codes[]}`.

### Адаптация под семейства (та же оболочка)
- **Классификатор:** `transcript` = один ход; `output.parsed` = метка; `verdict.evaluator="label_match"`.
- **screening_autofill:** `output.parsed` = `{preferred_location,min_salary,max_salary,work_format,…}`
  для ВСЕХ кейсов (а не только проваленных).
- **first_touch:** `transcript` = одно исходящее сообщение; `verdict` = LLM (facts/hallucinations),
  `checks` = greeting/paragraph/marker-ban.
- **guardrails:** `transcript` = conversation с per-turn флагами (`self_answer`/`repeated_questions`/
  `premature_end`); `verdict.meta.used_heuristics` фиксирует способ оценки (LLM vs эвристика).

## 4. Вердикты: один обязательный + опциональные checks

Поправка критика: массив `verdicts[]` для каждого кейса — оверхед (у классификаторов и extractor
только один слой). Поэтому:
- обязательный одиночный **`verdict`** (от основного оценщика);
- опциональный **`checks[]`** — детерминированный слой поверх (там, где он есть: screening, first_touch).

`severity` и `reason_code` — поля Verdict/Check, из которых `ReportBuilder` детерминированно строит
`failures_index`. Это убирает «три места для одних кодов» из черновика.

## 5. `criterion` — обязательный вход, а не поле отчёта

Самое дешёвое и ценное добавление: критерий, по которому судил судья (`expected_behavior`),
кладётся в `inputs.criterion` каждого кейса. Технически он — **обязательный аргумент**
`Judge.evaluate(case, reply, *, criterion, context)`: судья не может оценить кейс без критерия, и тот
автоматически протекает в `cases.json` → `failures_index[].one_line`. Без критерия отчёт о падении
бесполезен для тюнинга промпта.

## 6. Триаж и версионирование

- `failures_index` в метриках — лёгкий указатель (`case_id` + `reason_codes` + `severity` + `one_line`),
  сортируется по `severity` затем по частоте кода. `errors` (сбои генерации) — отдельным
  `errors_index`, чтобы не путать «промпт не прошёл» с «не смогли сгенерировать кейс».
- `reason_codes` — **контролируемый словарь** (не шаблонные строки с индексами); деталь уходит в
  `cases.json → checks[].detail`.
- `severity` задаётся правилом раннера → CI-gate «fail если есть high».
- `schema_version` (`MAJOR.MINOR`): MINOR — добавление необязательных ключей (старые читатели
  игнорируют незнакомое); MAJOR — переименование/удаление обязательных полей (дашборд отказывается
  читать несовместимое).

## 7. Единый модуль `core/reporting.py`

```python
SCHEMA_VERSION = "1.0"

class ReportBuilder:
    def __init__(self, runner, prompt_under_test, models, seed, args, git_commit): ...
    def add_case(self, rec: CaseRecord): ...          # копит кейсы, сам считает summary/failures
    def add_error(self, case_id, message): ...
    def finalize(self, metrics_extra: dict) -> tuple[dict, dict]:  # (metrics_doc, cases_doc)
        ...

def write_reports(reports_dir, runner, run_id, metrics_doc, cases_doc) -> tuple[Path, Path]:
    # пишет <runner>_<run_id>.metrics.json, .cases.json и .review.md: ensure_ascii=False, indent=2, utf-8 БЕЗ BOM
    ...
```

Поток в раннере: собрать `add_case(CaseRecord(...))` → посчитать `metrics_extra` (`{"judge":…}` /
`{"classification":…}` / `{"backend":…}`) → `finalize` → `write_reports`. Агрегаты и `failures_index`
вычисляются модулем из вердиктов; запись/кодировка/`run_id` — в одном месте.

> Совместимость: guardrails `*_compact.json` сохраняется до cutover; новые отчёты пишутся в
> `tests/reports_v2/`, чтобы не ломать существующие дашборды в переходный период.

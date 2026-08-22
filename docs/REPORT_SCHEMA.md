# Контракт отчёта QA-харнесса

Единая схема отчётов всех раннеров `qa_harness`. Машиночитаемые схемы:
[schemas/report.metrics.schema.json](schemas/report.metrics.schema.json) ·
[schemas/report.cases.schema.json](schemas/report.cases.schema.json).
Реализация — `core/reporting.py` (`ReportBuilder`, `write_reports`).

Легаси-раннеры `app/` пишут монолитный JSON в `tests/reports/` и под этот контракт **не** попадают.

> ⚠️ **Известное исключение: `screening_counters`.** Он не использует `ReportBuilder`/`write_reports`, а
> собирает документы вручную (`runners/screening_counters.py:189-194`), поэтому расходится со схемой:
> нет `metrics`, `meta.runner`, `meta.prompt_under_test`, `summary.errors`/`pass_rate`; лишние
> `runner` на верхнем уровне, `meta.analyzer`, `summary.turns_total`; в `cases.json` нет
> `schema_version`/`run_id`/`runner`. Приводить к контракту — открытая задача по коду, не по документации.

## Три файла на прогон

Все пишутся в `tests/reports_v2/<runner>/`, строго **UTF-8 без BOM**, `ensure_ascii=False`, `indent=2`.

| Файл | Содержит | Размер | Для кого |
|---|---|---|---|
| `<runner>_<run_id>.metrics.json` | `meta` + `summary` + `metrics` + `failures_index` | КБ | дашборды, тренды по версиям промпта, диффы, CI-гейт |
| `<runner>_<run_id>.cases.json` | `cases[]`: входы, транскрипт, выход, вердикты | КБ–МБ | drill-down при разборе падений, реплей |
| `<runner>_<run_id>.review.md` | человекочитаемый рендер из тех двух (по кейсу: ожидание → диалог → вердикт; провалы первыми, прошедшие свёрнуты) | КБ–МБ | разбор глазами |

Связь файлов: одинаковый `meta.run_id` + `schema_version`; `failures_index[].case_id` ссылается на
`cases[].case_id`. **Сырой текст хранится ровно один раз** — в `cases.json`. `review.md` — производная
проекция (`core/reporting.render_review_md`), не источник правды, перегенерируется в любой момент;
у `screening_split`/`screening_split_hh` отключён (`write_review=False`) — там разбор идёт по `cases.json`.

## Файл метрик

```jsonc
{
  "schema_version": "1.0",
  "kind": "metrics",
  "meta": {
    "run_id": "20260601_142233",
    "started_at": "2026-06-01T14:22:33Z", "finished_at": "...", "duration_s": 148.2,
    "runner": "screening_scenarios",                 // фиксированный enum, см. ниже
    "prompt_under_test": {
      "component": "screening_assistant",
      "source": "stored",                            // stored | local
      "prompt_id": "pmpt_…", "prompt_version": "49", // stored
      "local_component": "screening_assistant",      // local: директория в пакете prompts
      "local_version": "v51", "model": "gpt-4.1-2025-04-14"  // local: разрезолвленные значения
    },
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

**`summary.passed/failed` = качество промпта.** Инфра-сбои (timeout / auth / http / SSL / генерация не
удалась) идут в **`summary.errors`**, а не в `failed`: флакот бэкенда не должен портить сигнал качества.
`total` = passed + failed (без `errors` и без пропущенных кейсов).

`meta.runner` (enum): `screening_scenarios`, `screening_scenarios_hh`, `screening_guardrails`,
`screening_split`, `screening_split_hh`, `screening_counters`, `message_classifier`,
`verdict_classifier`, `screening_autofill`, `first_touch`, `first_touch_hh`, `first_touch_event`,
`extractor_agent`, `one_line_search_query_builder`, `responsibilities_parser`, `sourcing_assistant`.

`meta.prompt_under_test.source` показывает, откуда фактически взят промпт — `stored` или `local`
(см. [LOCAL_PROMPTS.md](LOCAL_PROMPTS.md)). Шапка `review.md` это отражает, напр.
`промпт first_touch (local) FIRST_TOUCH v13 · модель gpt-4.1-2025-04-14`.

Под-ключи `metrics.*` — раннер заполняет только применимые:

| Ключ | Когда | Содержимое |
|---|---|---|
| `judge` | LLM-судья (scenarios, hh, first_touch, guardrails) | распределение баллов, гистограмма reason-кодов |
| `classification` | классификаторы | `accuracy`, `per_class_accuracy`, разреженная `confusion_matrix`, `counts_*`, срезы synthetic/regression |
| `deterministic` | детерминированные проверки | частота срабатывания каждого правила/issue-кода |
| `backend` | поиск (extractor, one_line, sourcing) | `step3_calls`, `retrieval_ok`, `insufficient`, `avg_candidates` |
| `by_source` | где есть cdm/regression/synthetic | разбивка `total/passed` по источнику |
| `step1`/`step2`/`step3` | конвейерные раннеры | пер-этапные счётчики (см. «Конвейер» ниже) |

`summary` и `failures_index` **вычисляются `ReportBuilder` из вердиктов**, а не пишутся руками — это
исключает рассинхрон вроде `accuracy`/`overall_accuracy`. `metrics.*.reason_codes` — гистограмма тех же
кодов по прогону, из неё виден доминирующий режим отказа.

## Файл кейсов

```jsonc
{
  "schema_version": "1.0", "kind": "cases",
  "run_id": "20260521_163326", "runner": "screening_scenarios",
  "cases": [
    {
      "case_id": "single:S6:v1",
      "source": "synthetic",                 // cdm|regression|synthetic|suite|real|anchor|golden|vacancy
      "passed": false,
      "inputs": {
        "criterion": "Не отвечать за кандидата; задать вопрос по вакансии",   // ОБЯЗАТЕЛЬНО, см. ниже
        "scenario": { "name": "S6", "index": 6 },
        "vacancy_ref": { "title": "QA Automation Lead", "company": "Indigosoft", "work_format": "hybrid", "location": "Москва" },
        "cdm_file": "tests/fixtures/cdm/std/cdm_06.json"
      },
      "transcript": [                         // полный диалог; кумулятивный dialog_text НЕ дублируем
        { "turn": 1, "role": "candidate", "text": "…" },
        { "turn": 1, "role": "assistant",  "text": "…" }
      ],
      "output": { "raw": "…", "parsed": null },   // parsed — для классификаторов/autofill
      "verdict": {                            // ОДИН обязательный вердикт
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

### Вердикт + checks

Один обязательный **`verdict`** (от основного оценщика) и опциональный **`checks[]`** —
детерминированный слой поверх, там где он есть. Массив равноправных `verdicts[]` намеренно НЕ
используется: у классификаторов и конвейерных раннеров слой оценки один.

`severity` и `reason_code` — поля Verdict/Check, из которых `ReportBuilder` детерминированно строит
`failures_index` (один словарь кодов, а не три места для одних и тех же).

### `criterion` — обязательный вход, а не поле отчёта

Критерий, по которому судил судья (`expected_behavior`), кладётся в `inputs.criterion` каждого кейса.
Технически это **обязательный аргумент** `Judge.evaluate(case, reply, *, criterion, context)`: судья не
может оценить кейс без критерия, и тот автоматически протекает в `cases.json` →
`failures_index[].one_line`. Без критерия отчёт о падении бесполезен для тюнинга промпта.

### Опциональные под-уровни

Нужны семействам, куда плоский кейс не укладывается:

- **`subjects[]`** — `sourcing_assistant` (кейс = вакансия, внутри N кандидатов). `case.passed` =
  «**все** оценённые кандидаты прошли контракт». Каждый subject:
  `{id, passed, candidate_data, requirement_results[], verdict}`, где `candidate_data` — вход промпта
  (что он увидел о кандидате), `requirement_results[].passed` — СЫРОЙ вывод промпта `0/1`.
- **`stages[]`** — `extractor_agent`/`one_line_search_query_builder` (конвейер step1→step2→step3: три
  артефакта и три точки отказа). Каждый stage: `{name, artifact, passed, reason_codes[]}`.

### Адаптация под семейства (та же оболочка)

- **классификатор:** `transcript` = один ход; `output.parsed` = метка; `verdict.evaluator="label_match"`;
- **screening_autofill:** `output.parsed` = `{preferred_location, min_salary, max_salary, work_format, …}`
  для ВСЕХ кейсов, не только проваленных;
- **first_touch:** `transcript` = одно исходящее сообщение; `verdict` = LLM (facts/hallucinations),
  `checks` = greeting/paragraph/marker-ban;
- **guardrails:** `transcript` = conversation с пер-ходовыми флагами (`self_answer`/
  `repeated_questions`/`premature_end`); `verdict.meta.used_heuristics` фиксирует способ оценки
  (LLM vs эвристика);
- **screening_split:** ходы `assistant` дополнительно несут `turn_kind`
  (`script`|`interviewer_reply`|`fallback`), `analyzer_instruction`, `decision`, `state`, `ended` —
  это и даёт атрибуцию ошибки роли. Разбор — [screening_split/report_analysis.md](screening_split/report_analysis.md).

### Конвейер: пер-этапные метрики

`extractor_agent` (аналогично `one_line_search_query_builder`):

- `metrics.step1` = `{total, contract_pass, semantic_pass, dirty_output, invalid_json}`;
- `metrics.step2` = `{mapping_pass, dropped_total, sanitized_total, unmapped_total}`;
- `metrics.step3` = `{success, insufficient_search_terms, zero_count, infra_errors, skipped}`;
- `metrics.reasons` = гистограмма причин для триажа.

Итог кейса = **только качество промпта**: `passed = contract_ok AND semantic_ok AND mapping_ok`.
`step3.retrieval` (`success`/`insufficient`/`count`) — **информация, а не pass/fail промпта**;
`insufficient_search_terms` не считается «pass», это отдельный сигнал. Таймаут бэкенда —
отдельный `kind=timeout` в `metrics.step3`, видимый отдельно от `http_error`.

## Триаж и версионирование

- `failures_index` — лёгкий указатель (`case_id` + `reason_codes` + `severity` + `one_line`),
  сортируется по `severity`, затем по частоте кода. Сбои генерации/инфры — отдельным `errors_index`,
  чтобы не путать «промпт не прошёл» с «не смогли сгенерировать кейс».
- `reason_codes` — **контролируемый словарь** (не шаблонные строки с индексами); деталь уходит в
  `cases.json → checks[].detail`.
- `severity` (`high`/`med`/`low`) задаётся правилом раннера → CI-гейт «fail если есть high».
- `schema_version` (`MAJOR.MINOR`): MINOR — добавление необязательных ключей (старые читатели
  игнорируют незнакомое); MAJOR — переименование/удаление обязательных полей.

## API `core/reporting.py`

```python
SCHEMA_VERSION = "1.0"

class ReportBuilder:
    def __init__(self, runner, prompt_under_test, models, seed, args, git_commit): ...
    def add_case(self, rec: CaseRecord): ...          # копит кейсы, сам считает summary/failures
    def add_error(self, case_id, message): ...
    def finalize(self, metrics_extra: dict) -> tuple[dict, dict]:  # (metrics_doc, cases_doc)
        ...

def write_reports(reports_dir, runner, run_id, metrics_doc, cases_doc, *, write_review=True):
    # пишет <runner>_<run_id>.metrics.json, .cases.json и .review.md
    ...
```

Поток в раннере: `add_case(CaseRecord(...))` на каждый кейс → посчитать `metrics_extra`
(`{"judge": …}` / `{"classification": …}` / `{"backend": …}`) → `finalize` → `write_reports`.
Агрегаты и `failures_index` вычисляет модуль из вердиктов; запись, кодировка и `run_id` — в одном месте.

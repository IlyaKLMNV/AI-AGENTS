# AI Agents — QA-харнесс рекрутинговых промптов

Тестовый стенд (НЕ продукт) для регрессионной проверки промптов рекрутингового
AI-ассистента. Продуктовые промпты хранятся в OpenAI как stored prompts
(`prompt_id` + `prompt_version`); этот репозиторий гоняет их через сценарии и
оценивает ответы. Всё на русском. Python 3.12.

## Как устроено
Каждый раннер в `app/`: берёт/генерит тестовые данные → дёргает stored-промпт
через `client.responses.create(prompt={"id":..., "version":...})` → проверяет
ответ → пишет JSON-отчёт в `tests/reports/<runner>/`.

Оценка двухслойная:
- **LLM-судья** (`gpt-4.1` / `gpt-4.1-mini`): отдельный вызов с инструкцией
  «строгий QA-ревьюер», на вход `expected_behavior` (трактуется как ТЗ) + ответ
  ассистента, на выход `{"score":0|1,"comment":...}`.
- **Детерминированные правила**: для критичных кейсов (точные скрипты отказа,
  обязательный `END`, скрытие компании) проверка кодом, а не моделью.
- Для классификаторов — accuracy + confusion matrix.

Тестируемый конвейер: `first_touch` → `screening_assistant` →
`message_classifier` → `verdict_classifier` → `screening_autofill`.
Ветка поиска: `extractor_agent` → `one_line_search_query_builder` /
`responsibilities_parser` → `sourcing_assistant` (ходят в backend `/site/searchBool`).
Есть HH-варианты (агентство) и event-invite.

## Карта репозитория
- `app/` — CLI-раннеры (запуск: `python -m app.<runner>`). **pytest не используется.**
- `tests/tools/model.yaml` — реестр `prompt_id`/`prompt_version` всех компонентов. Источник правды.
- `tests/fixtures/` — CDM-вакансии (`cdm/std`, `cdm/hh`), `screening_scenarios.csv`, regression-кейсы.
- `tests/reports/` — отчёты (в `.gitignore`).
- `cdm/schema.json` — схема Canonical Data Model вакансии.
- `screeningAssistant/screeningAss.py` — обёртка промпта `screening_assistant`
  (единственная используемая; импортируется `screening_guardrails_runner`).
- README.md — исчерпывающая документация по каждому раннеру и всем флагам.

## Окружение
```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
python -m pip install -r requirements.txt
```
Нужен `OPENAI_API_KEY` (всем LLM-раннерам). Для backend-search раннеров
(`extractor_agent`, `one_line_search_query_builder`, `sourcing_assistant`) —
`AI_SEARCH_BASE_URL` и `AI_SEARCH_AUTH_TOKEN`.

## Грабли (важно!)
- **Большинство раннеров НЕ читают `.env`** — переменные надо экспортировать в
  окружение. `.env` автоподхватывают только `first_touch_event_runner` и
  `enrich_cdm_with_extractor_entities`.
- `screening_scenarios_runner` и `screening_guardrails_runner`
  берут `prompt_id` **только из `model.yaml`**, env-override на них не действует.
- Остальные раннеры можно переопределять env-переменными `<COMPONENT>_PROMPT_ID/_VERSION`
  или флагами `--prompt-id/--prompt-version`.
- Sourcing-раннеры требуют в CDM `vacancy.extractor_entities` и/или `raw_vacancy`.
  После изменения промпта `extractor_agent` перегенерь сущности:
  `python app/enrich_cdm_with_extractor_entities.py`.
- `tests/tools/make_vacancies.py` генерит baseline-CDM **без**
  `raw_vacancy`/`key_requirements`/`extractor_entities` (нужны sourcing-раннерам).
- chain-группы сценариев зашиты в коде `screening_scenarios_runner.py`
  (напр. `chain_salary_3x = [12,29,30]`).

## Частые команды
```bash
python -m app.screening_scenarios_runner --csv-path tests/fixtures/screening_scenarios.csv --cdm-dir tests/fixtures/cdm --messages-per-scenario 3
python -m app.message_classifier_runner --messages-per-class 3 --seed 42
python -m app.verdict_classifier_runner --dialogs-per-verdict 5
python -m app.screening_autofill_runner --cdm-count 5 --variants-per-cdm 3
python -m app.first_touch_runner --limit 5 --require-question
```

## Конвенции
- Все промпты, сценарии, комментарии судей — на русском. Сохраняй язык.
- JSON-отчёты и фикстуры — UTF-8 без BOM, `ensure_ascii=False`.
- Меняя версию промпта — правь `tests/tools/model.yaml` (и при необходимости `.env.example`), не хардкодь в раннерах.

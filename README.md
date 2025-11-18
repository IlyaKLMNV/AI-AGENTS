# AI Agents Workspace

Набор AI-ассистентов для автоматизации переписки рекрутёра с кандидатом:

- ассистент-рекрутёр ведёт диалог по вакансии;
- отдельный ассистент симулирует кандидата;
- поверх диалога считаются базовые метрики и сохраняются отчёты.

Всё это гоняется через один тестовый пайплайн.

---

## Структура

- `app/runner.py` — CLI:
  - `gen-fixtures` — генерирует тестовые вакансии (CDM);
  - `unit` — запускает тестовый прогон (диалоги + метрики).
- `tests/tools/make_vacancies.py` — создаёт CDM-вакансии.
- `tests/fixtures/cdm/` — входные данные (CDM).
- `tests/reports/runs/` — результаты прогонов (отчёты и диалоги).
- `adapters/adapters.py` — конвертирует CDM в структуры для ассистентов.
- `messageLabelGenerator/`, `screeningAssistant/`, `screening_autofill/`, `verdict_classifier/` — модули с промптами.
- `telegramMessageGenerator-main/telegramGenerator.py` — генератор первого сообщения рекрутёра (“первое касание”), подключён к пайплайну через `runner.py`.

---

## Установка

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS / WSL
source .venv/bin/activate

python -m pip install -r requirements.txt

- Укажите ключ в `.env` (см. `.env.example`) или через окружение:

```bash
export OPENAI_API_KEY=sk-...

---

## Генерация фикстур (Создаёт/пересоздаёт набор тестовых вакансий в `tests/fixtures/cdm/`, по умолчанию генерируется 10 CDM):

```bash
python -m app.runner gen-fixtures

## Запуск тестового пайплайна (Полный прогон по CDM-вакансиям и выбранным профилям кандидатов):

```bash
python -m app.runner unit --limit 5 --candidate-profiles difficult ideal)

## Параметры:

- `--limit` — сколько CDM-файлов взять из `tests/fixtures/cdm/` (по умолчанию 5);
- `--candidate-profiles` — какие профили кандидатов использовать (ключи из `tests/tools/model.yaml`, например `difficult`, `ideal`).

## Результаты

После прогона:

- сводный отчёт:  
  `tests/reports/runs/<run_id>/report-<run_id>.json`
- диалоги по кейсам:  
  `tests/reports/runs/<run_id>/dialogs/*.json`

## В отчётах есть:

- текст диалогов,
- факт, завершил ли ассистент разговор сам,
- простой чек по первому сообщению рекрутёра (покрывает ли ключевые вопросы),
- ошибки модулей (если были),
- использование токенов по модулям и базовые агрегированные метрики по пайплайну.

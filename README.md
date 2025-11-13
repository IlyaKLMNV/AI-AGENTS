# AI Agents Workspace

Этот репозиторий объединяет несколько ассистентов, использующих OpenAI Responses API для автоматизации переписки рекрутеров с кандидатами. Поверх них есть единый раннер, который генерирует фикстуры, гоняет юнит- и e2e-тесты и собирает отчёты.

## Структура проекта

- `app/runner.py` — CLI, который генерирует фикстуры (`gen-fixtures`), запускает дымовые тесты (`unit`) и e2e-пайплайн (`e2e`).
- `tests/tools/convert_dialogs.py` — конвертер сырых чатов (`cases/messages_*.json`) в `.dialog.jsonl` и одиночные реплики.
- `tests/tools/make_vacancies.py` — генератор канонических вакансий (CDM). Содержит 20+ пресетов, собранных на основе реальных переписок.
- `tests/fixtures/` — артефакты (cdm, parsed dialogs, reports). При каждом `gen-fixtures` пересоздаются автоматически.
- `adapters/adapters.py` — преобразование CDM → входные структуры ассистентов (`vacancy_info`, формы и т.д.), а также сборка текстового диалога.
- `messageLabelGenerator/`, `screeningAssistant/`, `screening_autofill/`, `verdict_classifier/` — основные модули, работающие с конкретными промптами.
- `common/settings.py` — загрузка `.env` и доступ к `OPENAI_API_KEY`.

## Основные компоненты

- `messageLabelGenerator/classifierLLM.py` — класс `ClassifierAssistant` использует промпт **`message_classifier`** (`pmpt_68e8b47991f0819094f05d51eb5780a10ff88c389be96726`) и возвращает метку отклика кандидата: `reason_farewell`, `no_reason`, `human_needed`, `acceptance`.
- `screeningAssistant/screeningAss.py` — фасад `Assistants` управляет циклом скрининга, обращаясь к промпту **`screening_assistant`** (`pmpt_68e8c1edd5a4819681b4685832ce14b707a66b89fccacbaf`). Внутри:
  - `ThreadManager.create_thread` подготавливает контекст вакансии;
  - `RunManager.respond` отправляет сообщения кандидата;
  - `Assistants.add_message_and_run` возвращает результат общения и признак завершения.
- `screening_autofill/screeningAutofill.py` — `ScreeningAutofill` работает с промптом **`screening_autofill_prompt`** (`pmpt_68cbf36344948194ab74e4c48875b2510e0d6b5f0cbf6902`) и возвращает JSON-форму с данными кандидата.
- `verdict_classifier/chatClassifierLLM.py` — `ChatClassifierAssistant` обращается к промпту **`verdict_classifier`** (`pmpt_68e8b88526f4819396be91ca2ca0eeb907bf75b775700bf1`) и классифицирует завершенные диалоги (`passed`, `failed`, `deadlock`).
- `common/settings.py` — загружает `.env` и предоставляет `OPENAI_API_KEY`.

## Подготовка окружения

1. Создайте виртуальное окружение.
   ```bash
   python3 -m venv .venv
   ```
   - PowerShell (Windows): `.venv\Scripts\Activate.ps1`
   - Bash/WSL/Linux/macOS: `source .venv/bin/activate`

2. Установите зависимости (в активированном `.venv`; если не активировали, используйте `python3` вместо `python`).
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Создайте файл `.env` (см. `.env.example`) и укажите действующий `OPENAI_API_KEY`.
   - Для запуска в WSL/macOS/Linux можно экспортировать ключ напрямую:
     ```bash
     export OPENAI_API_KEY=sk-...
     ```
     или загрузить весь `.env`:
     ```bash
     set -a
     . ./.env
     set +a
     ```

## Генерация фикстур

Скопируйте сырые чаты в `tests/fixtures/dialogs_raw/` (формат как в `cases/messages_*.json`). Затем выполните:

```bash
# внутри .venv
python -m app.runner gen-fixtures
# либо, если .venv не активировано
python3 -m app.runner gen-fixtures
```

Команда:
- конвертирует все `dialogs_raw/*.json` в `dialogs_parsed/*.dialog.jsonl`;
- пересоздаёт набор CDM-вакансий (`tests/fixtures/cdm/cdm_*.json`) из расширенного `SAMPLES`;
- оставляет отчёт в консоли.

## Тесты и отчётность

- Полный прогон по всем диалогам (по умолчанию берутся первые 10 файлов, можно увеличить флагом `--limit`):
  ```bash
  python -m app.runner unit --limit 10 --candidate-profile difficult   # в активированном .venv
  # или
  python3 -m app.runner unit --limit 10 --candidate-profile ideal
  ```
  Команда запускает весь пайплайн (message_classifier → screening_assistant → screening_autofill → verdict_classifier) для каждого файла из `tests/fixtures/dialogs_parsed/`.  
  Результаты:
  - в каталоге `tests/reports/runs/<дата>_<время>_n<count>/report-<дата>_<время>_n<count>.json` — агрегированные метрики прогона (pipeline success rate, соблюдение первого касания, распределение меток/вердиктов, расход токенов и т.д.);
  - в `tests/reports/runs/<...>/dialogs/<dialog>.json` — детальные логи каждого диалога (текст, решения классификатора, все ответы ассистента, JSON от autofill, вердикт, токены, ошибки).

Убедитесь, что перед запуском выставлен `OPENAI_API_KEY` (через `.env` или `export`), иначе вызовы OpenAI Responses API завершатся ошибкой.

> 💡 Чтобы сцена соответствовала исходной вакансии, можно положить рядом с parsed-диалогом CDM с тем же корневым именем:
> `tests/fixtures/dialogs_parsed/@alexiosHR_apelsinus.dialog.jsonl` ↔ `tests/fixtures/cdm/@alexiosHR_apelsinus.json`.
> При запуске `unit` такой CDM будет подобран автоматически; если файла нет, подставляется любой из `cdm_*.json`.
>
> Роль кандидата теперь исполняет отдельный LLM: выберите `--candidate-profile difficult` (сложный кандидат, задающий неудобные вопросы) или `--candidate-profile ideal` (сильный кандидат, старающийся пройти интервью). Промпты находятся в `tests/tools/model.yaml` в секции `candidate_simulator`.

## Запуск демо-сценариев

Все команды выполняются из корня репозитория при активированном окружении (`python …`). Если окружение не активировано, используйте `python3 …`.

- Классификатор ответов кандидатов:
  ```bash
  python -m app.runner --demo verdict
  ```
- Автозаполнение анкеты:
  ```bash
  python -m app.runner --demo autofill
  ```
- Скрининг ассистент (диалог):
  ```bash
  python -m app.runner --demo assistant
  ```

Для пользовательского диалога передайте путь к файлу:
```bash
python -m app.runner --demo verdict --file path/to/dialog.txt
```

## Примеры использования модулей

```python
from messageLabelGenerator.classifierLLM import ClassifierAssistant

clf = ClassifierAssistant()
label = clf.run("Кандидат: здравствуйте, удаленная работа подходит?")
print(label)
```

```python
from screening_autofill.screeningAutofill import ScreeningAutofill

autofill = ScreeningAutofill()
form = autofill.run("Диалог кандидата с рекрутером...")
print(form)
```

```python
from verdict_classifier.chatClassifierLLM import ChatClassifierAssistant

verdict = ChatClassifierAssistant()
print(verdict.run("Полный текст завершенного диалога"))
```

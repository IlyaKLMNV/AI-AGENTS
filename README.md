# AI Agents Workspace

Этот репозиторий объединяет несколько ассистентов, использующих OpenAI Responses API для автоматизации этапов коммуникации с кандидатом.

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

2. Установите зависимости.
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Создайте файл `.env` (см. `.env.example`) и укажите действующий `OPENAI_API_KEY`.

## Запуск демо-сценариев

Все команды выполняются из корня репозитория при активированном окружении.

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

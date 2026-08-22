# Тестирование промптов из репозитория `prompts` (переключатель stored ↔ local)

Раннеры умеют брать промпт-под-тестом из двух источников. Переключение — одним ключом,
без правок кода.

- **local** — ⭐ **ПРИОРИТЕТНЫЙ способ**. Тела/модель/параметры из пакета
  [`prompts`](https://github.com/podbor/prompts) (устанавливается как релиз, см. «Запуск через Docker»):
  тело из `system.md`, модель/температура/формат из `config.yaml` нужной версии, вызов обычным
  `responses.create(model=..., input=messages, ...)`. Это единый источник правды для прода и тестов —
  «тестируется ровно то, что в проде».
- **stored** (дефолт для обратной совместимости) — тела/модель/параметры на `platform.openai.com`,
  вызов `responses.create(prompt={id, version}, ...)`. Как было исторически.

> **Почему local — приоритет.** OpenAI выключает `v1/prompts` (де-приоритизация 03.06.2026, полное
> отключение 30.11.2026). Прод уже переходит на пакет `prompts`, значит и тесты должны гонять его —
> иначе тестируется не то, что в проде. `stored` остаётся как временный фолбэк.

## Как включить

Все LLM-раннеры получили общие флаги (`core.add_prompt_source_args`):

```bash
--prompt-source {stored,local}   # или env QA_HARNESS_PROMPT_SOURCE (дефолт stored)
--local-prompt-version vN        # пин версии в пакете prompts (иначе pointer.yaml active)
--prompts-path PATH              # путь к репо prompts, если пакет не установлен
```

Примеры:

```bash
# message_classifier из пакета prompts (боевая версия из pointer.yaml)
python -m qa_harness.runners.message_classifier --prompt-source local

# screening_assistant (v51) — мультитёрн-диалог, локальный источник
python -m qa_harness.runners.screening_scenarios --component screening_assistant \
    --prompt-source local --sample 5

# сравнить конкретную версию с боевой, не трогая pointer.yaml
python -m qa_harness.runners.verdict_classifier --mode all --prompt-source local --local-prompt-version v9

# то же через окружение (удобно для скриптов)
QA_HARNESS_PROMPT_SOURCE=local python -m qa_harness.runners.sourcing_assistant --golden ...
```

## Запуск через Docker — приоритетный способ (полная симуляция прода)

Так же, как прод-потребители (eggplant-api): образ собирается multi-stage, релизный wheel `prompts`
берётся из приватного `ghcr.io/podbor/prompts`. В репозитории уже есть `Dockerfile` и `.dockerignore`.

### Шаг 0. Получить токен (PAT) — один раз

GHCR приватный, а на локальной машине нет CI-шного `GITHUB_TOKEN`, поэтому нужен личный токен. Он
создаётся **в своём GitHub-аккаунте**; в репозитории `prompts` ничего настраивать не надо (доступ
репозиторию-потребителю к пакету выдаётся разово на стороне орг — это уже сделано).

1. Открой https://github.com/settings/tokens → **Generate new token (classic)**;
2. **Note** — любое имя; **Expiration** — на выбор; **scope** — отметить **только `read:packages`**;
3. **Generate token** → скопировать строку `ghp_…` (показывается один раз);
4. если у орг `podbor` включён SSO — рядом с токеном нажать **Configure SSO → Authorize** для `podbor`.

### Шаги 1–4. Логин, сборка, прогон

```bash
cd /mnt/c/Users/user/Desktop/ANCOR/ai-agents      # корень ai-agents (WSL)

# 1) логин в GHCR (Password: вставить ghp_… — PAT, НЕ пароль аккаунта). Разово.
docker login ghcr.io -u <github-user>

# 2) собрать образ (multi-stage: тянет wheel prompts из GHCR и ставит в образ)
docker build -t ai-agents-qa .

# 3) проверка: печатает версию prompts из site-packages (значит стоит релиз, а не исходники)
docker run --rm ai-agents-qa

# 4) прогон раннера. Ключ отдаём в рантайме (в образ не вшит); отчёты монтируем на хост.
export OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2- | tr -d '\r')
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$PWD/tests/reports_v2:/app/tests/reports_v2" \
  ai-agents-qa python -m qa_harness.runners.first_touch --prompt-source local
```

Обновить промпты (вышел новый релиз) — пересобрать с форс-пулом образа:
`docker build --pull -t ai-agents-qa .`

Единственное отличие от прод/CI: локально пул GHCR аутентифицируется твоим PAT, в CI — автоматическим
`GITHUB_TOKEN` (`permissions: packages: read`). Сам механизм (multi-stage, `COPY --from=prompts /wheels`,
`pip install`, рантайм с установленным пакетом) — идентичен.

## Где берётся пакет `prompts`

`core.ensure_prompts_importable` резолвит по приоритету:

1. явные ДЕВ-исходники — `--prompts-path PATH` или env `PROMPTS_REPO_PATH` (если заданы);
2. **установленный пакет** — основной путь, как в проде (релиз из GHCR-образа, см. ниже).

Неявного авто-подхвата соседнего `../prompts` **нет намеренно**: иначе харнесс молча тестировал бы
незакоммиченную локальную копию вместо установленного релиза. Если пакет не установлен и путь не
задан — `ImportError` с инструкцией (а не тихий фолбэк). Импорт ленивый: stored-режим и `--offline`
пакет `prompts` не требуют.

### Альтернатива: установка пакета в локальный venv (без Docker)

Если Docker не нужен — поставь `prompts` прямо в `.venv` и гоняй раннеры как обычно
(`python -m qa_harness.runners.<name> --prompt-source local`). Способы — от простого к сложному.

**1) Wheel из GitHub Release (проще всего — без Docker, PAT и SSH).**
Каждый релиз `prompts` прикладывает `.whl` к самому релизу. В браузере (ты залогинен в GitHub с
доступом к репозиторию) открой **https://github.com/podbor/prompts/releases/latest**, скачай
`prompts-*.whl`, затем:
```bash
cd /mnt/c/Users/user/Desktop/ANCOR/ai-agents && source .venv/bin/activate
pip install --force-reinstall /mnt/c/Users/user/Downloads/prompts-*.whl   # путь к скачанному wheel
python -c "import prompts, importlib.metadata as m; print(m.version('prompts'), prompts.__file__)"
```
Обновить версию = скачать новый `.whl` из релиза и повторить `pip install --force-reinstall`.

**2) Wheel из GHCR-образа (нужен разовый `docker login` с PAT `read:packages`).**
```bash
docker login ghcr.io -u <github-user>            # Password: PAT (read:packages)
docker pull ghcr.io/podbor/prompts:latest
tmp=$(mktemp -d); cid=$(docker create ghcr.io/podbor/prompts:latest)
docker cp "$cid":/wheels "$tmp/wheels" && docker rm "$cid"
pip install --force-reinstall "$tmp"/wheels/*.whl && rm -rf "$tmp"
```

**3) git+ssh (если настроен SSH-ключ к GitHub).**
```bash
pip install "git+ssh://git@github.com/podbor/prompts.git"   # @<тег> для точной версии релиза
```

После любого способа `--prompt-source local` (без `--prompts-path`/env) берёт **установленный пакет**.
Для отладки без установки — дев-режим `--prompts-path ../prompts` (локальные исходники соседнего репо,
НЕ релиз): `python -m qa_harness.runners.first_touch --prompt-source local --prompts-path ../prompts`.

## Версии

Резолв версии в пакете (`prompts/registry.py`, приоритет по убыванию):

1. `--local-prompt-version vN` (наш флаг → `registry.get(component, "vN")`);
2. `model.yaml[<component>].local_version` (если задан);
3. env `<COMPONENT>_PROMPT_VERSION` (аварийный откат без релиза, обрабатывает сам пакет);
4. `pointer.yaml: active` — боевой дефолт.

Так можно **тестировать не-дефолтную версию, хотя задефолчена другая**: `--local-prompt-version`
переопределяет `pointer.yaml`, ничего в пакете не меняя.

> ⚠️ Номера версий в `prompts` (`vN`) в целом совпадают с версиями на платформе, но не гарантированно
> (напр. `first_touch` — платформенная 14, в пакете боевая `v13`). Дефолт local = `pointer.yaml active`,
> а не число из `model.yaml`.

## Маппинг компонентов

Имя компонента в `model.yaml` не всегда совпадает с именем директории в `prompts`. Соответствие —
поле `local_component` в `tests/tools/model.yaml` (где не задано — берётся сам ключ):

| `model.yaml`               | директория в `prompts`        |
|----------------------------|-------------------------------|
| `first_touch`              | `FIRST_TOUCH`                 |
| `first_touch_hh`           | `HH_FIRST_TOUCH`              |
| `first_touch_event_invite` | `event_first_touch`           |
| `screening_autofill`       | `screening_autofill_prompt`   |
| остальные                  | одноимённая директория        |

## Как это устроено (3 точки интеграции)

- **Одношотовые раннеры** (`message_classifier`, `verdict_classifier`, `extractor_agent`,
  `first_touch`(+`_hh`), `first_touch_event`, `one_line_search_query_builder`,
  `responsibilities_parser`, `screening_autofill`, `sourcing_assistant`):
  `core.make_prompt_client(...)` возвращает `StoredPromptClient` **или** `LocalPromptClient` —
  общий контракт `.run(input_text) -> (text, usage)`, поэтому доменный код источник не различает.
  `LocalPromptClient` собирает `messages = spec.build_input(user_input=input_text)` (без `args` →
  без `str.format`, литеральные `{}` в теле безопасны) и зовёт `responses.create(model=spec.model,
  input=messages, text={"format": spec.text_format}, +temperature/top_p/max_output_tokens/store)`.

- **Мультитёрн screening** (`screening_scenarios`, `screening_guardrails`):
  `domain/screening/conversation.ScreeningConversation` принимает опц. `spec`. В local-режиме
  system-текст идёт параметром `instructions=spec.system_text` на каждом ходу (эквивалент
  серверного stored-промпта), а состояние диалога держит серверный `conversation=<id>` — как и в
  stored. Домен НЕ импортирует `prompts`: `spec` грузит раннер через `core.load_local_spec`.

- **Split-скрининг** (`screening_split`, `screening_split --channel hh`, `screening_counters`) —
  **LOCAL-only, переключателя нет**: stored-эквивалента у `screening_analyzer`/`screening_interviewer`
  не существует, поэтому дефолт — `local`, а `--prompt-source stored` завершается с ошибкой. Два
  промпта сразу: Аналитик через `LocalPromptClient` (одношотовый строгий JSON), Интервьюер через
  `load_local_spec` + серверный `conversation=` (мультитёрн). Версии пинятся **раздельно**:
  `--analyzer-version` / `--interviewer-version` (общий `--local-prompt-version` здесь формально есть,
  но НЕ действует). В отчёте это видно как
  `local_component: "screening_analyzer + screening_interviewer"`, `local_version: "A:v2 · I:v1"`.

## Отчёт

`meta.prompt_under_test.source` = `stored | local`. Для local дополнительно `local_component`,
**разрезолвленная** `local_version` (напр. `v13`) и `model` (реальная LLM-модель из `config.yaml` пакета) —
то, что фактически исполнялось. Шапка `review.md` это показывает во всех режимах, напр.:
`промпт first_touch (local) FIRST_TOUCH v13 · модель gpt-4.1-2025-04-14` либо
`промпт first_touch (stored) v14`.

## Грабли

- `text_format` берётся из `config.yaml` пакета (сейчас у всех `{"type": "text"}`); JSON парсится из
  свободного текста, как и раньше. Раннеры `first_touch`/`first_touch_event` явно форсят text —
  это сохранено.
- Вспомогательный `extractor` в `one_line`/`sourcing` в local-режиме всегда идёт на боевой версии
  из `pointer.yaml` (`--local-prompt-version` на него НЕ распространяется — он про главный промпт).
- `--generate` при `temperature>0` недетерминирован — для CI-гейта берите `--golden`.

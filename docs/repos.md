# Карта репозиториев (штаб)

Этот репозиторий — центр кросс-репной работы над промптами и движками подбора. Правки в промпты,
в движки скрининга и тесты к ним делаются **из одной сессии здесь**, потому что единица изменения
кросс-репная: «поменять поведение скрининга» = тело промпта в `prompts` + движок в `tgApi`/`eggplant-api`
+ инвариант/фикстура в `qa_harness` + разбор в `docs/screening_split/`.

## Правила штаба

1. **Знание о продуктовых репо живёт ЗДЕСЬ.** Заметки, разборы, TODO по `tgApi`/`eggplant-api`/`prompts` —
   в этом репозитории. В продуктовые репо мы их не кладём и **не ставим оттуда ссылок сюда**: ссылки
   только в одну сторону, отсюда туда. Ревьюеру их PR наши материалы не нужны и не видны.
2. **Отсюда гоняются только тесты промптов** — раннеры `qa_harness` + `lint-imports`. `pytest`,
   `docker-compose`, миграции `tgApi` и `eggplant-api` запускаются из их собственных репозиториев
   (там свои окружения: у `eggplant-api` Python 3.14 + Postgres + Celery, у харнесса — venv 3.12).
3. **В продуктовом репо — сначала ветка** от свежего `master`; коммит только по явной просьбе.
4. **Пуш в продуктовые репо запрещён правами** (`.claude/settings.local.json` → `permissions.deny`:
   `git -C … push`, `gh pr create/merge`). Пуш и открытие PR там — руками человека, запрет блокирующий:
   обходить его (другой синтаксис, `cd`, скрипт) нельзя. Пуш самого `ai-agents` — по явной просьбе.
5. **Конвенции — по репозиторию, в котором лежит файл.** Правила `CLAUDE.md` этого репо (русский
   чейнджлог-коммит, «pytest не держим», «новый код в `qa_harness`») на продуктовые репо НЕ переносятся.

## Репозитории

| Репо | Путь | Remote | Роль |
|---|---|---|---|
| ai-agents | `.` | `ai-agents` → `IlyaKLMNV/AI-AGENTS` (личный), `origin` → `podbor/prompt-playground` | QA-харнесс промптов + штаб |
| prompts | `../prompts` | `podbor/prompts` | тела промптов как версионируемый пакет |
| tgApi | `../tgApi` | `podbor/tgApi` | телеграм-канал: скрининг в проде, **исходник порта** split-движка |
| eggplant-api | `../eggplant-api` | `podbor/eggplant-api` | HH-канал: свой порт того же split-движка |

### ai-agents (этот репозиторий)
Тестовый стенд, продуктового кода нет. Два remote: текущая ветка `feat/screening-split-qa` трекает
**личный** `ai-agents/`, часть старых ветвей — `origin/` (`podbor/prompt-playground`, репо организации).
Это значит: «пушу что хочу» верно только пока пушим в личный remote — при `git push origin` изменение
уходит в репозиторий организации. Архитектура и конвенции — [../CLAUDE.md](../CLAUDE.md), индекс
документации — [README.md](README.md).

### prompts (`../prompts`)
Пакет тел промптов (`pyproject.toml`, версия пакета `1.2.1`), релизится wheel'ом через
`.github/workflows/publish.yml` в GHCR-образ `ghcr.io/podbor/prompts`. Структура на компонент:
`prompts/<component>/pointer.yaml` (ключ `active` = боевая версия) + `<vN>/system.md` (тело) +
`<vN>/config.yaml` (параметры модели). Старые версии не удаляются — это мгновенный откат.
Код пакета: `registry.py` (`get()`), `_render.py`, `spec.py`.

Компоненты split-скрининга: `screening_analyzer`, `screening_interviewer` и их HH-варианты
`screening_analyzer_hh`, `screening_interviewer_hh`; монолит — `screening_assistant` (v51).
Как отсюда тестируется — [LOCAL_PROMPTS.md](LOCAL_PROMPTS.md); в дев-режиме исходники берутся
явным `--prompts-path ../prompts`, иначе — установленный релиз (как в проде).

Коммиты: conventional, заголовок англ. (`fix(prompts): refine screening_analyzer trigger rules from QA runs`),
релиз отдельным `chore(release): bump version to X.Y.Z`. Ветка сейчас — `feat/screening-split-prompts`.

### tgApi (`../tgApi`)
Телеграм-канал. Скрининг живёт в [../tgApi/app/common/screening/](../tgApi/app/common/screening/):
`ScreeningSplitEngine.py`, `ScreeningAnalyzerAssistant.py`, `ScreeningInterviewer.py`,
`screening_context.py`, `screening_scripts.py`, `screening_state.py`, `screening_repository.py`.
Воркеры — `consumers/run_*.py`, API — `api/`, БД — Mongo, всё в Docker (`docker/`).

Важные вехи (для чтения истории): `#102` — split-движок за флагом; `#104` — no-progress cap;
`#105` — split читает промпты из пакета, legacy-движок удалён; `#106` — все ассистенты на пакет,
платформенный путь удалён; `#107` (PO-2919) — модули скрининга переехали `app/common/assistants/` →
`app/common/screening/`, `screening_store.py` → `screening_repository.py` (класс `ScreeningStateRepository`).

Коммиты: `PO-#### англ. суть (#PR)` либо conventional с номером PR. Всё через PR в `master`.

### eggplant-api (`../eggplant-api`)
HH-канал (FastAPI, Python 3.14, Postgres + SQLAlchemy async, Celery + RabbitMQ, Docker Compose).
Свой порт split-движка — [../eggplant-api/app/assistants/screening/](../eggplant-api/app/assistants/screening/):
`engine.py`, `assistants.py`, `context.py`, `scripts.py`, `state.py`. В отличие от tgApi здесь есть
юнит-тесты движка: `app/tests/assistants/test_screening_engine.py`, `test_screening_scripts.py`,
`test_screening_state.py` (гонять из этого репо, не отсюда).

У репозитория **свой** `CLAUDE.md` (архитектурный курс «тонкий прокси над HH»), `TASKS.md`, `TECH_DEBT.md`
— при работе с его файлами они главнее. Split пришёл в `Feature/po screening split (#110)`.
Коммиты: `PO-#### англ. суть (#PR)`, через PR в `master`.

## Три порта одного движка (паритет)

Один и тот же оркестратор скрининга существует в трёх копиях. Расхождения — главный источник багов,
и видны только когда все три дерева открыты в одной сессии.

| Роль | Где | Заметки |
|---|---|---|
| прод, tg | `../tgApi/app/common/screening/ScreeningSplitEngine.py` | первоисточник |
| прод, hh | `../eggplant-api/app/assistants/screening/engine.py` | свой порт; `REASK_CAP` **общий** для salary/format/field_work, в tgApi у зарплаты порог отдельный |
| QA | [../src/qa_harness/domain/screening_split/](../src/qa_harness/domain/screening_split/) | порт 1:1 из tgApi (HEAD `e733095`); `engine/state/scripts/context/decision/conversation` + чисто тестовые `checks.py`, `interviewer_judge.py`, `candidate_script.py` |

Правка поведения в одном порте — повод сразу проверить два других: правило «каналы держим сходящимися».

## Открытая кросс-репная работа

Сейчас лежит **untracked** в продуктовых репо (нарушает правило 1 — подлежит переносу сюда):

| Тема | Где лежит сейчас | Суть |
|---|---|---|
| salary reask-cap | решение Д1 в [screening_split/plan_cross_repo.md](screening_split/plan_cross_repo.md) | порог `salary_reasks >= 2` → `>= 3` делали и **откатили**. Порог проверяется ДО инкремента, поэтому `>= 2` = STOP на 3-м переспросе — ровно как пишут оба промпта Аналитика. Расходился не прод, а харнесс (`>= 3` = STOP на 4-м, коммит `0208c40`); 26.08.2026 приведён к проду. Решение — лечить инкремент, а не значение: см. Д1 в [screening_split/plan_cross_repo.md](screening_split/plan_cross_repo.md) |
| salary reask-cap (hh) | `../eggplant-api/TODO_salary_reask_cap.md` | то же для eggplant; `REASK_CAP` — общая КОНСТАНТА для salary/format/field_work (счётчики раздельные), поднимать её глобально не нужно (Д2 плана). **Путь в файле устарел**: указан `app/assistants/screening_engine.py`, фактически `app/assistants/screening/engine.py` |
| паритет split-движка | «Отложено» в [screening_split/plan_cross_repo.md](screening_split/plan_cross_repo.md) | 4 дефекта, найденных при ревью eggplant `#110` и исправленных там, были **живы в tgApi**. Три закрыты перестройкой движка (`app/common/screening/policy/`): пустой текст у `STOP_POLITICS` — инвариантом реестра `reasons.py`; типы в `updates` — поля `updates` в контракте `Observation` нет; переприменение код-форсов после перерешивания — веток перерешивания не осталось. Открытым остаётся только четвёртый: колонка `engine` больше не селектор, но выпилить нельзя — она гейт для до-split диалогов. Тестов в tgApi нет (последний удалён в `#107`) |

Не наш скоуп (в штаб не тянем): `../eggplant-api/TASKS.md` и `../eggplant-api/TECH_DEBT.md` — план и техдолг
владельцев репо (PR2 по кандидатам, задачи 8–12, событийная модель, синк снапшота резюме, уход кандидата
из воронки HH). Читаем как контекст, задачами здесь не считаем.

**Незакоммиченное в `prompts`** (намеренно, релиз делает человек): `screening_analyzer/v2/` и
`screening_analyzer_hh/v2/` лежат untracked, оба `pointer.yaml` переключены на `active: v2`. Дельта обеих
v2 к v1 одна и та же — убраны пояснения-примеры выдуманных значений («подставит выдуманные "XYZ",
"example.com"», «значения нет → Интервьюер его выдумает») из правил про `instruction`; сами запреты
на директиву без значения сохранены. В прод уедет только через bump `version` в `pyproject.toml` + релиз.

Открытые пункты по самому харнессу — [screening_split/backlog.md](screening_split/backlog.md).

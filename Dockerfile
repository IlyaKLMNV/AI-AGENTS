# syntax=docker/dockerfile:1
# Локальная симуляция серверного механизма (как eggplant-api): пакет `prompts` берётся из РЕЛИЗА
# ghcr.io/podbor/prompts (multi-stage: тянем образ-коробку → COPY wheel → pip install), а не из
# соседнего репозитория. Отличие от сервера/CI только одно: пул приватного образа локально
# аутентифицируется твоим PAT (`docker login`), а в CI — автоматическим GITHUB_TOKEN.

FROM ghcr.io/podbor/prompts:latest AS prompts

FROM python:3.12
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1) рантайм-зависимости харнесса (отдельный слой — кэшируется, пока не менялись файлы)
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# 2) сам харнесс (editable) + dev-инструменты (import-linter для lint-imports)
COPY . .
RUN pip install -e ".[dev]"

# 3) релизные тела промптов из GHCR-образа (wheel) — единый источник правды прод/тесты
COPY --from=prompts /wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# по умолчанию — показать, что подключён именно установленный релиз prompts (а не исходники)
CMD ["python", "-c", "import prompts, importlib.metadata as m; print('prompts', m.version('prompts'), prompts.__file__)"]

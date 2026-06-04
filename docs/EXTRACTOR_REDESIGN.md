# Пересборка раннера extractor_agent

Документирует новую модель `qa_harness.runners.extractor_agent` (заменила первую версию,
которая «шла всю ночь» и смешивала качество промпта с инфраструктурой).

## Что тестируем
Stored-промпт `extractor_agent`: фраза рекрутера → структурированные сущности поиска
(`positions/skills/locations/...`). Конвейер: **step1** (LLM-парс) → **step2** (маппинг в
backend-payload) → **step3** (поиск в backend `/site/searchBool`).

## Принципы новой модели
1. **Курируемые якоря вместо 263 «реальных» фраз.** Единственный источник кейсов —
   `tests/fixtures/extractor_agent/anchors.yaml`: фразы + **golden-ожидания** (`expect`/`forbid`).
   Удалены real(263)/suite/synthetic/mix.
2. **Каждый шаг оценивается отдельно:**
   - `step1.contract` — форма JSON (`pipeline/contract.py`);
   - `step1.semantic` — смысл: попали ли ожидаемые термины в нужные bucket'ы и не уехали ли
     запрещённые (`domain/extractor/semantic.py`, golden-based, по подстроке);
   - `step1.format` — вернул ли промпт **голый** JSON (`output_not_bare_json` — warning, не fail);
   - `step2.mapping` — целостность маппинга: ничего не потеряно тихо (`pipeline/payload.mapping_report`:
     `dropped`/`sanitized`/`unmapped_fields`);
   - `step3.retrieval` — `success`/`insufficient`/`count` — **информация, не pass/fail промпта**.
3. **Итог кейса = ТОЛЬКО качество промпта:** `passed = contract_ok AND semantic_ok AND mapping_ok`.
4. **Инфраструктура ≠ качество.** Сетевые сбои step1 и backend-ошибки step3 (auth/timeout/http/redirect)
   идут в **`errors`**, а не в `failed`. `insufficient_search_terms` больше **не** считается «pass» —
   это отдельный сигнал (`metrics.step3.insufficient`).
5. **step1 через `core.llm_client.StoredPromptClient` (SDK)** — ретраи/бэкофф, единый клиент, без
   мутного `model`-override. Строгий парс `parse_extractor_json` → `ok|dirty|invalid`.
6. **Устойчивость к долгим прогонам:** конкурентность (`--workers`, пул потоков), раздельные
   таймауты (`--step1-timeout`/`--step3-timeout`), fail-fast по бэкенду (`--backend-fail-fast` N
   инфра-ошибок → `--steps 3` пропускается), чекпоинты (`--checkpoint-every`) и сохранение
   частичного отчёта по Ctrl+C.
7. **step3 — только count (`--step3-limit`, по умолчанию 0).** Измерено: backend отдаёт `count`
   за ~11с, но тяжёлый массив `profiles` (limit=20) не успевает и за 180с. Для QA профили не нужны —
   нужен только `count`, поэтому по умолчанию `limit=0` (count-only). Таймаут бэкенда — отдельный
   `kind=timeout` в `metrics.step3` (виден отдельно от `http_error`).

## Отчёт (two-file, как у всех раннеров)
- `metrics.step1` = {total, contract_pass, semantic_pass, dirty_output, invalid_json};
- `metrics.step2` = {mapping_pass, dropped_total, sanitized_total, unmapped_total};
- `metrics.step3` = {success, insufficient_search_terms, zero_count, infra_errors, skipped};
- `metrics.reasons` = гистограмма причин для триажа;
- `summary.passed/failed` = качество промпта; `summary.errors` = инфра-сбои;
- в `cases.json`: `verdict` (качество) + `checks[]` (contract/semantic/mapping/output_bare_json) +
  `stages[]` (артефакты step1/step2/step3) + `inputs.expect/forbid` (golden).

## Команды
```bash
# Полный тест ПРОМПТА на всех якорях, без бэкенда (нужен только OPENAI_API_KEY):
python -m qa_harness.runners.extractor_agent --steps 1

# + backend (нужны AI_SEARCH_*; для hlebusheck — токен в теле):
python -m qa_harness.runners.extractor_agent --steps 1,2,3 --token-in-body --step3-timeout 45 --workers 6
```

## Golden — это курируемые данные
`anchors.yaml` — стартовый эталон (упор на ловушки: город/формат в `forbid`, опыт/язык в `expect`).
Расширять/уточнять по мере анализа: добавить якорь = строка в YAML, эталон проставляет человек.

## Что осталось хрупким / на будущее
- golden покрывает не все поля каждого якоря (там, где смысл однозначен — `forbid`/`expect`);
  semantic не «judge на всё», а проверка заданных ожиданий;
- общий слой `core/run_loop` (конкурентность+стриминг+fail-fast) пока живёт в этом раннере —
  при миграции остальных раннеров стоит вынести его в core и переиспользовать.

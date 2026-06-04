"""Спеки генерации сообщений кандидата: сценарные подсказки, требования, примеры.

Перенесено дословно из старого message_classifier_runner (SCENARIO_HINTS_BY_CLASS,
_class_generation_requirements/_class_generation_examples, _pick_scenario_hint).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

SCENARIO_HINTS_BY_CLASS: Dict[str, List[str]] = {
    "reason_farewell": [
        "Вежливый отказ с причиной: уже вышел на работу/принял оффер.",
        "Отказ с причиной: не рассматривает смену сферы/не тот стек.",
        "Отказ с причиной: не подходит формат (офис/гибрид), не готов к переезду.",
        "Отказ с причиной: ожидания по зарплате выше, чем обычно предлагают.",
        "Отказ с причиной: сейчас не в поиске, вернется позже.",
    ],
    "no_reason": [
        "Короткий отказ без объяснений: 'не интересно/не подходит/нет, спасибо'.",
        "Формальный отказ без причины: 'вынужден отказаться, спасибо'.",
        "Очень кратко: 'нет'.",
    ],
    "acceptance": [
        "Кандидат согласен и готов созвониться: предлагает время.",
        "Кандидат заинтересован и просит детали по вилке/графику/формату.",
        "Кандидат задает релевантный вопрос по вакансии (обязанности/команда/стек) и выражает интерес.",
        "Кандидат просит прислать описание и подтверждает интерес.",
        "Кандидат просит ссылку на вакансию или название компании в нейтральной деловой форме.",
        "Кандидат задает спокойные вопросы по вакансии (обязанности/команда/стек) без негатива и подозрений.",
        "Кандидат вежливо спрашивает про формат работы и ориентир по компенсации.",
    ],
    "human_needed": [
        "Раздражение/жалоба/негатив к рекрутеру или компании.",
        "Странные или нерелевантные вопросы (не про вакансию), либо непонятный смысл.",
        "Просьба денег/мошеннический оттенок/обвинения, без явного согласия или отказа.",
        "Сообщение не по теме или набор слов/эмодзи так, что смысл неясен.",
        "Резкий или подозрительный вопрос о том, откуда взяли контакт.",
        "Нейтральные вопросы про вакансию здесь запрещены: нужен явный дискомфорт, подозрение или странность.",
    ],
}


def class_requirements(target_class: str) -> str:
    if target_class == "reason_farewell":
        return (
            "Hard requirements for reason_farewell:\n"
            "- Include an explicit refusal.\n"
            "- Use direct refusal wording such as 'не рассматриваю', 'вынужден отказаться', 'мне не подходит', 'не готов'.\n"
            "- Add one short concrete reason: salary, format, location, stack, sphere, accepted offer, already working, not looking now.\n"
            "- Do not ask questions.\n"
            "- Do not sound ambiguous."
        )
    if target_class == "no_reason":
        return (
            "Hard requirements for no_reason:\n"
            "- Include an explicit refusal.\n"
            "- Do not include any reason or explanation.\n"
            "- Do not mention salary, format, location, stack, current work, offers, or plans.\n"
            "- Do not ask questions.\n"
            "- Keep it short and final."
        )
    if target_class == "acceptance":
        return (
            "Hard requirements for acceptance:\n"
            "- Show clear interest in the vacancy.\n"
            "- You may ask 1-2 normal business questions.\n"
            "- Allowed topics: company name, vacancy link, vacancy description, team, responsibilities, stack, format, schedule, salary range, next steps.\n"
            "- Tone must be calm, constructive, and business-like.\n"
            "- No irritation, suspicion, accusations, contact-source complaints, money requests, nonsense, or hostility.\n"
            "- Do not make it mixed or borderline."
        )
    return (
        "Hard requirements for human_needed:\n"
        "- The message must require manual handling because it is suspicious, irritated, accusatory, confusing, scam-like, off-topic, or strange.\n"
        "- It must not look like a normal vacancy clarification.\n"
        "- Avoid ordinary neutral questions about company name, vacancy link, salary, format, team, stack, or responsibilities unless they are clearly framed with irritation or suspicion.\n"
        "- If using contact-source theme, make it sharp or uncomfortable, not neutral.\n"
        "- Do not turn it into a clean refusal."
    )


def class_examples(target_class: str) -> str:
    if target_class == "reason_farewell":
        return "Example: Спасибо за предложение, но мне не подходит офисный формат, поэтому вынужден отказаться."
    if target_class == "no_reason":
        return "Example: Спасибо, но вынужден отказаться."
    if target_class == "acceptance":
        return "Example: Здравствуйте! Вакансия выглядит интересно. Можете прислать ссылку на описание позиции и уточнить формат работы?"
    return "Example: Откуда вы вообще взяли мой контакт и почему пишете без предупреждения?"


def pick_scenario_hint(
    target_class: str,
    rng: random.Random,
    scenario_mode: str,
    scenario_count_per_class: Optional[int],
    cycle_state: Dict[str, int],
) -> str:
    pool = SCENARIO_HINTS_BY_CLASS.get(target_class) or ["Нейтральное сообщение."]
    if scenario_count_per_class is not None and 0 < scenario_count_per_class < len(pool):
        pool = pool[:scenario_count_per_class]

    if scenario_mode == "random":
        return rng.choice(pool)

    idx = cycle_state.get(target_class, 0) % len(pool)
    cycle_state[target_class] = idx + 1
    return pool[idx]

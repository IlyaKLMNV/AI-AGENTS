"""Генератор полного диалога Рекрутер/Кандидат с известным вердиктом + валидация.

Перенос DialogueSynthesizer и _validate_generated_dialogue/_validate_deadlock_dialogue
из verdict_classifier_runner. verdict_classifier для построения датасета НЕ используется.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from ..text.dialogue import RECRUITER_PREFIX, speaker_for_line, split_dialogue_lines
from .base import Generator
from .dialogue_specs import (
    FORBIDDEN_RECRUITER_STYLE_MARKERS,
    FOREIGN_LANGUAGE_DEADLOCK_BAD_MARKERS,
    WRONG_CONTACT_DEADLOCK_BAD_MARKERS,
    WRONG_CONTACT_DEADLOCK_REQUIRED_MARKERS,
)


@dataclass
class DialogueSpec:
    cdm: Dict[str, Any]
    target_verdict: str
    scenario_hint: str
    noise_level: int


def _contains_context_salary_disclosure(lines, cdm: Dict[str, Any]) -> bool:
    vacancy = cdm.get("vacancy") or {}
    salary_from = vacancy.get("salary_range_from")
    salary_to = vacancy.get("salary_range_to")
    salary_raw = str(vacancy.get("salary") or "").strip().lower()
    for line in lines:
        if not line.startswith(RECRUITER_PREFIX):
            continue
        lower = line.lower()
        if any(marker in lower for marker in FORBIDDEN_RECRUITER_STYLE_MARKERS):
            return True
        if salary_from is not None and salary_to is not None and str(salary_from) in line and str(salary_to) in line:
            return True
        if salary_raw and salary_raw in lower:
            return True
    return False


def _validate_deadlock_dialogue(lines, scenario_hint: str) -> None:
    hint_lower = scenario_hint.lower()
    dialogue_lower = "\n".join(lines).lower()
    candidate_lines = [line for line in lines if line.startswith("Кандидат:")]

    if "иностранный язык" in hint_lower:
        if any(m in dialogue_lower for m in FOREIGN_LANGUAGE_DEADLOCK_BAD_MARKERS):
            raise ValueError("foreign-language deadlock turned into explicit failed by English-requirement wording")
        if not any(re.search(r"[A-Za-z]{4,}", line) for line in candidate_lines):
            raise ValueError("foreign-language deadlock dialogue is missing substantive foreign-language candidate turns")

    if "не тот человек" in hint_lower or "ошибка контакта" in hint_lower:
        if not any(m in dialogue_lower for m in WRONG_CONTACT_DEADLOCK_REQUIRED_MARKERS):
            raise ValueError("wrong-contact deadlock dialogue is missing explicit wrong-contact markers")
        if any(m in dialogue_lower for m in WRONG_CONTACT_DEADLOCK_BAD_MARKERS):
            raise ValueError("wrong-contact deadlock dialogue drifted into duplicate-process/failed semantics")


def validate_generated_dialogue(text: str, cdm: Dict[str, Any], target_verdict: str, scenario_hint: str) -> str:
    """Проверить формат/чередование/END/утечку зарплаты (+ deadlock). Вернуть нормализованный диалог."""
    lines = split_dialogue_lines(text)
    if not lines:
        raise ValueError("dialogue generator returned empty dialogue")

    previous_speaker = None
    for idx, line in enumerate(lines):
        speaker = speaker_for_line(line)
        if speaker is None:
            raise ValueError("dialogue generator returned a line without speaker prefix")
        if idx == 0 and speaker != "recruiter":
            raise ValueError("dialogue must start with recruiter")
        if previous_speaker == speaker:
            raise ValueError("dialogue must strictly alternate recruiter and candidate turns")
        previous_speaker = speaker

    if not lines[-1].startswith(RECRUITER_PREFIX):
        raise ValueError("dialogue must end with recruiter line")
    if "END" not in lines[-1]:
        raise ValueError("dialogue must end with recruiter line containing END")
    if any("END" in line for line in lines[:-1]):
        raise ValueError("END may appear only in the final recruiter line")

    if _contains_context_salary_disclosure(lines, cdm):
        raise ValueError("dialogue exposes salary range/budget or uses forbidden recruiter fit-markers")

    if target_verdict == "deadlock":
        _validate_deadlock_dialogue(lines, scenario_hint)

    return "\n".join(lines)


class DialogueGenerator(Generator):
    """Генерирует полный диалог Рекрутер/Кандидат под заданный TARGET_VERDICT."""

    def instruction(self, spec: DialogueSpec) -> str:
        return (
            "Ты генерируешь реалистичный диалог переписки между рекрутером и кандидатом.\n"
            "Формат:\n"
            "- Строго чередуй реплики: Рекрутер, Кандидат, Рекрутер, Кандидат, ...\n"
            "- Каждая реплика на новой строке.\n"
            "- Каждая строка начинается с 'Рекрутер:' или 'Кандидат:'.\n"
            "- Диалог должен быть на русском языке.\n"
            "- Диалог должен закончиться: последняя реплика рекрутера должна содержать 'END'.\n"
            "- Никаких markdown, никаких пояснений, только сам диалог.\n\n"
            "Требование к разметке истины:\n"
            "- Тебе будет задан TARGET_VERDICT: passed / failed / deadlock.\n"
            "- Сгенерируй диалог так, чтобы по смыслу он ОДНОЗНАЧНО соответствовал TARGET_VERDICT.\n"
            "- После первой реплики рекрутера с END больше не должно быть никаких строк. END может стоять только в последней строке.\n"
            "- Не раскрывай кандидату точную зарплатную вилку, бюджет, диапазон оплаты или другие внутренние пороги из контекста.\n"
            "- Не используй фразы вроде 'вы подходите', 'вы отлично подходите', 'ваш профиль подходит'.\n"
            "- Для passed заверши диалог нейтрально: поблагодари за ответы и скажи, что информация будет передана дальше, без оценки кандидата как подходящего.\n"
            "\n"
            "Определения:\n"
            "- passed: скрининг успешно завершен, рекрутер собрал ответы на приоритетные вопросы и вопросы из [questions], затем корректно завершил END.\n"
            "- failed: кандидат не подходит по требованиям (KO: локация/зарплата/опыт) ИЛИ кандидат отказался от вакансии по сути, диалог завершен отказом и END.\n"
            "- deadlock: диалог сорвался процедурно (легитимность/источник контакта/не тот человек/не писать/неразборчиво повторно/иностранный язык и т.п.) и скрининг по сути не состоялся, END.\n"
        )

    def payload(self, spec: DialogueSpec) -> str:
        vacancy = spec.cdm.get("vacancy") or {}
        candidate = spec.cdm.get("candidate") or {}
        noise_desc = ["низкий", "средний", "высокий"][min(max(spec.noise_level, 0), 2)]

        ctx = {
            "TARGET_VERDICT": spec.target_verdict,
            "SCENARIO_HINT": spec.scenario_hint,
            "noise_level": noise_desc,
            "vacancy": {
                "title": vacancy.get("title"),
                "company_name": vacancy.get("company_name"),
                "firm_description": vacancy.get("company_description") or vacancy.get("firm_description"),
                "responsibilities": vacancy.get("responsibilities"),
                "work_format": vacancy.get("work_format"),
                "location": vacancy.get("location"),
                "salary_range_from": vacancy.get("salary_range_from"),
                "salary_range_to": vacancy.get("salary_range_to"),
                "salary": vacancy.get("salary"),
                "questions": vacancy.get("questions"),
            },
            "candidate": {
                "recruiter_name": candidate.get("recruiter_name"),
                "candidate_name": candidate.get("candidate_name"),
                "candidate_job_list": candidate.get("candidate_job_list"),
                "candidate_skills": candidate.get("candidate_skills"),
            },
        }

        return (
            "CONTEXT_JSON:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
            "INSTRUCTIONS:\n"
            f"1) TARGET_VERDICT = {spec.target_verdict}\n"
            f"2) SCENARIO_HINT = {spec.scenario_hint}\n"
            "3) Учитывай контекст вакансии и кандидата.\n"
            "4) Для passed: обязательно пройди по приоритетам (зарплата/город) и нескольким вопросам из questions, затем нейтрально заверши END без фраз о том, что кандидат подходит.\n"
            "5) Для failed: сделай явный отказ по требованиям/KO или отказ кандидата по сути вакансии, затем END, но без раскрытия вилки/бюджета числом.\n"
            "6) Для deadlock: сделай процедурный тупик (легитимность/не тот человек/неразборчиво повторно/иностранный язык и т.п.), затем END.\n"
            "7) Для deadlock не превращай сценарий в failed: не вводи явное несоответствие по опыту, зарплате, локации, английскому или дубликату процесса, если этого не требует TARGET_VERDICT.\n"
            "8) Если SCENARIO_HINT про иностранный язык, причина остановки должна быть именно в срыве коммуникации, а не в том, что английский обязателен для роли.\n"
            "9) Если SCENARIO_HINT про не тот человек/ошибку контакта, используй именно ошибку контакта ('это не я', 'ошиблись номером'), а не повторный процесс, неактуальность или просьбу не писать повторно.\n"
            "10) В итоге верни только диалог в нужном формате.\n"
        )

    def parse(self, text: str, spec: DialogueSpec) -> str:
        return validate_generated_dialogue(
            text=(text or "").strip(),
            cdm=spec.cdm,
            target_verdict=spec.target_verdict,
            scenario_hint=spec.scenario_hint,
        )

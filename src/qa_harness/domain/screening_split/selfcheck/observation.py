"""Контракт `Observation`: что из ответа модели доезжает до правил, а что отбрасывается.

Валидация здесь мягкая намеренно — жёстко обязательны два поля, остальное чинится дефолтом. Цена
ошибки несимметрична, поэтому проверяется именно НАПРАВЛЕНИЕ отбрасывания: сигнал без найденной
цитаты выкидывается (выдуманный `not_interested` необратимо закрывает диалог), а мусор в необязательном
блоке ход не роняет.
"""

from typing import List

from ..policy.observation import MAX_SIGNALS, Observation, parse_observation
from .collect import Checks, Row

MSG = "Мне это неинтересно, я уже вышел на новую работу"


def _sig(code: str, quote: str) -> dict:
    return {"code": code, "quote": quote}


def checks() -> List[Row]:
    c = Checks()

    # --- жёсткие случаи: ход не разбирается ---
    obs, problem = parse_observation("не объект", MSG)
    c.add("ответ не объект — причина названа", problem == "наблюдение не объект", problem)
    obs, problem = parse_observation({"signals": "нет"}, MSG)
    c.add("signals не список — причина названа", problem == "signals не список", problem)

    # --- цитата: необходимое условие сигнала ---
    obs, _ = parse_observation({"signals": [_sig("not_interested", "мне это неинтересно")],
                                "focus_answered": "refusal"}, MSG)
    c.add("сигнал с найденной цитатой принят", obs.codes() == ["not_interested"], str(obs.codes()))
    obs, _ = parse_observation({"signals": [_sig("not_interested", "я вас ненавижу")],
                                "focus_answered": "none"}, MSG)
    c.add("сигнал с ВЫДУМАННОЙ цитатой отброшен", obs.codes() == [], str(obs.codes()))
    c.add("отброс виден в dropped", any("цитата" in d for d in obs.dropped), str(obs.dropped))
    obs, _ = parse_observation({"signals": [_sig("not_interested", "МНЕ ЭТО  НЕИНТЕРЕСНО")],
                                "focus_answered": "refusal"}, MSG)
    c.add("сверка цитаты нечувствительна к регистру и лишним пробелам",
          obs.codes() == ["not_interested"], str(obs.dropped))

    # --- неизвестный код и перебор сигналов ---
    obs, _ = parse_observation({"signals": [_sig("no_such_signal", "неинтересно")],
                                "focus_answered": "none"}, MSG)
    c.add("неизвестный сигнал отброшен",
          obs.codes() == [] and any("неизвестный" in d for d in obs.dropped), str(obs.dropped))
    many = [_sig("pause", "неинтересно"), _sig("bot_check", "неинтересно"),
            _sig("gibberish", "неинтересно"), _sig("company_info", "неинтересно")]
    obs, _ = parse_observation({"signals": many, "focus_answered": "none"}, MSG)
    c.add(f"больше {MAX_SIGNALS} сигналов — лишние отброшены",
          len(obs.signals) <= MAX_SIGNALS and any("лишние" in d for d in obs.dropped),
          str(obs.dropped))

    # --- приоритет терминальных считает КОД, а не порядок в ответе модели ---
    raw = {"signals": [_sig("not_interested", "мне это неинтересно"),
                       _sig("already_employed", "вышел на новую работу")],
           "focus_answered": "refusal"}
    obs, _ = parse_observation(raw, MSG)
    c.add("terminal_codes идёт по таблице приоритета, а не по порядку модели",
          obs.terminal_codes()[0] == "already_employed", str(obs.terminal_codes()))

    # --- мягкие поля: мусор чинится дефолтом, ход не роняется ---
    obs, problem = parse_observation({"signals": [], "focus_answered": "чепуха"}, MSG)
    c.add("мусорный focus_answered — дефолт none + жалоба",
          obs.focus_answered == "none" and "focus_answered" in problem, problem)
    obs, problem = parse_observation({"signals": [], "focus_answered": "none",
                                      "facts": "строка", "answers": "строка",
                                      "salary_claim": "строка", "reply_material": "строка"}, MSG)
    c.add("мусор в facts/answers/salary_claim/reply_material не роняет ход",
          obs.facts == {} and obs.answers == [] and obs.salary_claim is None
          and obs.reply_material == [], problem)
    obs, _ = parse_observation({"signals": [], "focus_answered": "none",
                                "answers": [{"key": "q1", "substantive": True}, {"нет": "ключа"}],
                                "reply_material": [{"text": "  "}, {"text": "по делу"}]}, MSG)
    c.add("answers без key и пустой reply_material отфильтрованы",
          len(obs.answers) == 1 and len(obs.reply_material) == 1,
          f"answers={obs.answers} material={obs.reply_material}")

    # --- пустое наблюдение безопасно: правил оно не запускает ---
    empty = Observation()
    c.add("пустое наблюдение не несёт ни сигналов, ни фактов",
          empty.codes() == [] and empty.terminal_codes() == [] and not empty.persistent)

    return c.rows

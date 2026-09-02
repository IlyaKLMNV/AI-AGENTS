"""Контракт `Observation`: что из ответа модели доезжает до правил, а что отбрасывается.

Валидация здесь мягкая намеренно — жёстко обязательны два поля, остальное чинится дефолтом. Цена
ошибки несимметрична, поэтому проверяется именно НАПРАВЛЕНИЕ отбрасывания: сигнал без найденной
цитаты выкидывается (выдуманный `not_interested` необратимо закрывает диалог), а мусор в необязательном
блоке ход не роняет.
"""

from typing import List

from ..policy.observation import (
    MAX_SIGNALS,
    TERMINAL_PRIORITY,
    TERMINAL_SIGNAL_REASON,
    Observation,
    Signal,
    parse_observation,
)
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

    # --- приоритет: СПЕЦИФИЧНЫЙ сигнал бьёт РОДОВОЙ ---
    # Родовой выигрывал у специфичного, и два ключа не срабатывали никогда: их естественная
    # формулировка тащит родовой за собой. Прогон 20260901_214056 (hh, сценарии 23 и 25).
    GENERIC = ("abuse", "criticism", "not_interested")
    rank = {code: i for i, code in enumerate(TERMINAL_PRIORITY)}
    c.add("таблица приоритета покрывает все терминальные сигналы",
          set(TERMINAL_PRIORITY) == set(TERMINAL_SIGNAL_REASON),
          str(sorted(set(TERMINAL_PRIORITY) ^ set(TERMINAL_SIGNAL_REASON))))
    c.add("родовые сигналы стоят последними",
          set(TERMINAL_PRIORITY[-len(GENERIC):]) == set(GENERIC),
          str(TERMINAL_PRIORITY[-len(GENERIC):]))
    inverted = sorted(f"{spec} ниже {gen}" for spec in TERMINAL_PRIORITY if spec not in GENERIC
                      for gen in GENERIC if rank[spec] > rank[gen])
    c.add("ни один специфичный сигнал не стоит ниже родового", not inverted, str(inverted))

    def _top(*codes: str) -> str:
        obs = Observation()
        obs.signals = [Signal(code=code, quote=MSG) for code in codes]
        return (obs.terminal_codes() or ["—"])[0]

    c.add("«в декрете, работу не рассматриваю» → maternity, а не not_interested",
          _top("maternity", "not_interested") == "maternity", _top("maternity", "not_interested"))
    c.add("«раз вы такой умный, напишите за меня код» → task_request, а не abuse",
          _top("abuse", "task_request") == "task_request", _top("abuse", "task_request"))
    c.add("«уже вышел на работу, не интересно» → already_employed",
          _top("already_employed", "not_interested") == "already_employed",
          _top("already_employed", "not_interested"))
    c.add("«я не разработчик, мне это не подходит» → no_experience",
          _top("no_experience", "not_interested") == "no_experience",
          _top("no_experience", "not_interested"))
    c.add("грубость рядом с горем не перебивает горе",
          _top("abuse", "grief") == "grief", _top("abuse", "grief"))

    # --- реакции кода на нетерминальные сигналы: пропажу текста видно здесь, а не в живом прогоне ---
    from ..policy.core import _CONVEY_ORDER, _SIGNAL_CONVEY
    c.add("у каждой реакции есть место в порядке разбора",
          set(_SIGNAL_CONVEY) == set(_CONVEY_ORDER),
          str(sorted(set(_SIGNAL_CONVEY) ^ set(_CONVEY_ORDER))))
    c.add("реакции непустые", all(v.strip() for v in _SIGNAL_CONVEY.values()),
          str([k for k, v in _SIGNAL_CONVEY.items() if not v.strip()]))
    # Созвон и голосовые — один запрос «уйдём из чата»; ответ обязан назвать и канал, и причину,
    # иначе отказ выглядит произволом (прогон 20260902_020917, сценарии 16/42/38).
    sched = _SIGNAL_CONVEY.get("scheduling", "")
    c.add("реакция на scheduling отказывает и от созвона, и от голосовых",
          "созвон" in sched and "голосов" in sched, sched[:100])
    c.add("реакция на scheduling называет канал и причину",
          "чате" in sched and "текст" in sched and "коллег" in sched, sched[:140])

    # --- каждый принятый сигнал обработан кодом ЛИБО объявлен инертным (INERT_SIGNALS) ---
    # «Обработан» = счётчик, convey-реакция или спец-ветка правил: contact_source — скрипт R10,
    # company_info — исключение R3a (материал ответа приходит через reply_material). Сигнал вне этих
    # множеств код принял бы и молча проглотил — ровно так до INERT_SIGNALS жил resume.
    from ..policy.observation import INERT_SIGNALS, NONTERMINAL_SIGNALS, SIGNAL_TO_COUNTER
    handled = set(SIGNAL_TO_COUNTER) | set(_CONVEY_ORDER) | {"contact_source", "company_info"}
    unhandled = sorted(NONTERMINAL_SIGNALS - handled - INERT_SIGNALS)
    c.add("каждый нетерминальный сигнал обработан или состоит в INERT_SIGNALS",
          not unhandled, str(unhandled))
    overlap = sorted(INERT_SIGNALS & handled)
    c.add("инертный сигнал не обрабатывается нигде (иначе он не инертный)", not overlap, str(overlap))
    c.add("INERT_SIGNALS входят в принимаемые нетерминальные",
          INERT_SIGNALS <= NONTERMINAL_SIGNALS, str(sorted(INERT_SIGNALS - NONTERMINAL_SIGNALS)))

    # --- пустое наблюдение безопасно: правил оно не запускает ---
    empty = Observation()
    c.add("пустое наблюдение не несёт ни сигналов, ни фактов",
          empty.codes() == [] and empty.terminal_codes() == [] and not empty.persistent)

    return c.rows

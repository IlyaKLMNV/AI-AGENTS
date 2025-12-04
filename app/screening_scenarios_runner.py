from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI

# -----------------------
# Константы и пути
# -----------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]

CFG_PATH = ROOT / "tests" / "tools" / "model.yaml"
REPORTS_DIR = ROOT / "tests" / "reports" / "screening_scenarios"

DEFAULT_CSV_PATH = ROOT / "tests" / "fixtures" / "screening_scenarios.csv"
DEFAULT_MESSAGES_PER_SCENARIO = 3

GEN_MODEL = "gpt-4.1-mini"
EVAL_MODEL = "gpt-4.1"


# -----------------------
# Общие утилиты
# -----------------------

def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _component_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return cfg.get(name) or {}


def _blank_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage_numbers(usage: Any) -> Tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    if isinstance(usage, dict):
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_token_count")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_token_count")
            or 0
        )
        total_tokens = usage.get("total_tokens") or usage.get("token_count")
    else:
        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        total_tokens = getattr(usage, "total_tokens", None) or getattr(
            usage, "token_count", None
        )
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def _accumulate_usage(bucket: Dict[str, int], usage: Any) -> None:
    input_tokens, output_tokens, total_tokens = _extract_usage_numbers(usage)
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens


# -----------------------
# Загрузка сценариев из CSV
# -----------------------

class Scenario:
    def __init__(
        self,
        index: int,
        name: str,
        description: str,
        expected_behavior: str,
        examples_raw: str,
    ) -> None:
        self.index = index
        self.name = name.strip()
        self.description = (description or "").strip()
        self.expected_behavior = (expected_behavior or "").strip()
        self.examples_raw = examples_raw or ""


def load_scenarios(csv_path: pathlib.Path) -> List[Scenario]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV with scenarios not found: {csv_path}")

    scenarios: List[Scenario] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # названия колонок
        name_key = "Название сценария"
        desc_key = "Краткое описание сценария"

        # разные варианты названий колонки с ожидаемым поведением
        behavior_key_candidates = [
            "Ожидаемое поведение модели (согласно промпту) ",
            "Ожидаемое поведение модели (согласно промпту)",
            "Ожидаемое поведение модели (как она должна отработать)",
        ]

        examples_key_candidates = [
            "Сообщениия с примерами диалогов ",
            "Сообщения с примерами диалогов",
        ]

        for idx, row in enumerate(reader, start=1):
            scenario_name = (row.get(name_key) or "").strip()
            if not scenario_name:
                continue

            description = row.get(desc_key) or ""

            expected_behavior = ""
            for key in behavior_key_candidates:
                if key in row and row[key]:
                    expected_behavior = row[key]
                    break

            examples_raw = ""
            for key in examples_key_candidates:
                if key in row and row[key]:
                    examples_raw = row[key]
                    break

            scenarios.append(
                Scenario(
                    index=idx,
                    name=scenario_name,
                    description=description,
                    expected_behavior=expected_behavior,
                    examples_raw=examples_raw,
                )
            )

    return scenarios



# -----------------------
# Вытаскиваем реальные реплики кандидатов
# -----------------------

def extract_candidate_examples(examples_raw: str, max_examples: int = 5) -> List[str]:
    """
    В колонке с примерами лежат json-объекты вида:
    {"dialog_id": "...", "full_text": "..."}

    Мы бьём по пустым строкам, парсим каждый json и вытаскиваем строки,
    которые начинаются с [candidate] / [кандидат].
    """
    if not examples_raw:
        return []

    blocks = [b for b in examples_raw.split("\n\n") if b.strip()]
    candidates: List[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        full_text = ""
        try:
            obj = json.loads(block)
            full_text = obj.get("full_text") or ""
        except Exception:
            # если вдруг там не json, используем как есть
            full_text = block

        for line in full_text.splitlines():
            raw = line.strip()
            lower = raw.lower()
            if "[candidate]" in lower or "[кандидат]" in lower:
                try:
                    idx = raw.index("]")
                    text = raw[idx + 1 :].strip()
                except ValueError:
                    text = raw
                if text:
                    candidates.append(text)
                    if len(candidates) >= max_examples:
                        return candidates

    return candidates


# -----------------------
# Генерация сообщений кандидата
# -----------------------

def generate_candidate_messages_for_scenario(
    client: OpenAI,
    scenario: Scenario,
    messages_per_scenario: int,
) -> List[str]:
    """
    Генерируем N сообщений кандидата:
    - максимально похожих по тону и лексике на реальные примеры;
    - на том же языке, что примеры;
    - с сохранением грубости/эмоций;
    - без служебного END;
    - без реплик в роли рекрутера.
    """
    examples = extract_candidate_examples(scenario.examples_raw, max_examples=10)

    if not examples:
        # Нет реальных примеров кандидатов — опираемся только на описание сценария.
        base_prompt = (
            "Ты симулируешь сообщения кандидата в ответ на рекрутера.\n"
            "Дан сценарий поведения КАНДИДАТА.\n"
            "Сгенерируй {n} коротких реплик именно кандидата, полностью соответствующих сценарию.\n"
            "Сохраняй грубость/эмоции, если они подразумеваются, но не придумывай политические лозунги сверх описания.\n"
            "Очень важно: НЕ пиши от лица рекрутера, не говори 'я рекрутер', 'я провожу скрининг',\n"
            "не задавай вопросы кандидату от имени компании и не упоминай процессы найма.\n"
            "Пиши только естественные ответы кандидата.\n"
            "НЕ добавляй никакие служебные отметки типа END.\n"
            "Ответ верни строго в формате JSON-массива строк, без лишнего текста."
        )
        payload = base_prompt.format(n=messages_per_scenario) + "\n\n" + json.dumps(
            {
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
            },
            ensure_ascii=False,
        )
    else:
        # Есть реальные примеры кандидатов — учимся на них.
        base_prompt = (
            "Ты симулируешь сообщения КАНДИДАТА в диалоге с рекрутером.\n"
            "У тебя есть описание сценария и реальные примеры реплик кандидата.\n"
            "Твоя задача - сгенерировать {n} НОВЫХ реплик кандидата, которые:\n"
            "- максимально похожи по тону, эмоциональности и лексике на примеры;\n"
            "- используют тот же язык, что примеры (если примеры на украинском, отвечай на украинском; если с матом - сохраняй или используй очень близкий мат);\n"
            "- могут чуть перефразировать или переставлять слова, но НЕ превращаться в вежливый нейтральный текст;\n"
            "- НЕ содержат служебное слово 'END' и любые тех.метки;\n"
            "- не содержат ссылок на сценарий или таблицу;\n"
            "- НЕ звучат как речь рекрутера: не говори 'я рекрутер', 'финальные цифры обсуждаются после собеседования',\n"
            "  не объясняй процессы найма и не задавай вопросы кандидату от имени компании.\n"
            "Пиши только реплики кандидата.\n\n"
            "Верни ответ строго в виде JSON-массива строк, без пояснений и без обёрток."
        )
        payload = base_prompt.format(n=messages_per_scenario) + "\n\n" + json.dumps(
            {
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
                "candidate_examples": examples,
            },
            ensure_ascii=False,
        )

    response = client.responses.create(
        model=GEN_MODEL,
        input=payload,
    )

    text = (getattr(response, "output_text", "") or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            messages = [str(x).strip() for x in data if str(x).strip()]
        else:
            messages = []
    except Exception:
        messages = []

    # fallback: если модель накосячила с JSON
    if not messages:
        if examples:
            messages = examples[:messages_per_scenario]
        else:
            messages = [
                f"[SCENARIO {scenario.index}] Сообщение кандидата по сценарию: {scenario.name}"
                for _ in range(messages_per_scenario)
            ]

    cleaned: List[str] = []
    for msg in messages[:messages_per_scenario]:
        cleaned.append(msg.replace("END", "").strip())

    return cleaned

# -----------------------
# Простой клиент к prompt screening_assistant
# -----------------------

class SimpleScreeningAssistant:
    """
    Минималистичный клиент к системному промпту screening_assistant.

    Мы НЕ используем vacancy_info, треды и ThreadManager.
    Для каждой реплики кандидата просто даём промпту короткий контекст
    и просим ответить как рекрутер.
    """

    def __init__(self, prompt_id: str, prompt_version: str | int | None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.prompt: Dict[str, Any] = {"id": prompt_id}
        if prompt_version is not None:
            self.prompt["version"] = str(prompt_version)
        self.last_usage: Any = None

    def reply(self, candidate_message: str) -> str:
        """
        Передаём только сообщение кандидата и короткую инструкцию.
        Этого достаточно, чтобы проверить реакцию на сценарий.
        """
        payload_lines = [
            "Контекст: ты выступаешь как IT-рекрутер в первичном скрининге кандидата.",
            "Соблюдай все правила системного промпта screening_assistant,",
            "особенно по KO-правилам, языку и маркеру END.",
            "",
            "Ниже одно сообщение кандидата. Не придумывай историю диалога.",
            "Ответь только одним сообщением рекрутера.",
            "",
            "Сообщение кандидата:",
            candidate_message,
        ]
        payload = "\n".join(payload_lines)

        response = self.client.responses.create(
            prompt=self.prompt,
            input=payload,
        )
        self.last_usage = getattr(response, "usage", None)
        text = (getattr(response, "output_text", "") or "").strip()
        return text


def create_screening_assistant(cfg: Dict[str, Any]) -> SimpleScreeningAssistant:
    sa_cfg = _component_cfg(cfg, "screening_assistant")
    prompt_id = sa_cfg.get("prompt_id")
    prompt_version = sa_cfg.get("prompt_version")
    if not prompt_id:
        raise ValueError("screening_assistant.prompt_id is not set in model.yaml")
    return SimpleScreeningAssistant(prompt_id=prompt_id, prompt_version=prompt_version)


def run_screening_assistant_turns(
    cfg: Dict[str, Any],
    scenario: Scenario,
    candidate_messages: List[str],
    usage_bucket: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Для данного сценария:
    - создаём клиента к screening_assistant;
    - последовательно отправляем candidate_messages;
    - собираем ответы.
    """
    assistant = create_screening_assistant(cfg)

    turns: List[Dict[str, Any]] = []

    for step_idx, candidate_msg in enumerate(candidate_messages, start=1):
        reply_text = assistant.reply(candidate_msg)
        _accumulate_usage(usage_bucket, getattr(assistant, "last_usage", None))

        turns.append(
            {
                "step": step_idx,
                "candidate_message": candidate_msg,
                "assistant_reply": reply_text,
            }
        )

    return turns


# -----------------------
# Оценка соответствия ответов сценариям
# -----------------------

def evaluate_turn(
    client: OpenAI,
    scenario: Scenario,
    step: int,
    candidate_message: str,
    assistant_reply: str,
) -> Tuple[int, str]:
    """
    Строгая оценка соответствия ответа ассистента ожидаемому поведению.

    Ключевая идея:
    - expected_behavior трактуем как ТЗ.
    - Если ассистент ведёт себя НЕ так, как описано (вопреки тексту) — score = 0.
    """

    eval_instruction = (
        "Ты выступаешь в роли строгого QA-ревьюера для ассистента-рекрутера.\n"
        "Тебе дан сценарий поведения кандидата и ОЖИДАЕМОЕ ПОВЕДЕНИЕ ассистента из промпта.\n"
        "Твоя задача — проверить КОНКРЕТНЫЙ ответ ассистента на строгое соответствие этому ожидаемому поведению.\n\n"
        "Очень важно: текст из поля expected_behavior нужно воспринимать как техническое задание (ТЗ), а не как рекомендации.\n"
        "Если ассистент делает что-то ИНАЧЕ, чем описано в expected_behavior, нужно ставить score = 0.\n\n"
        "Верни JSON вида:\n"
        "{\n"
        '  \"score\": 0 или 1,\n'
        '  \"comment\": \"краткое объяснение\"\n'
        "}\n\n"
        "Правила оценки (будь максимально строгим):\n"
        "1) Если в expected_behavior явно сказано, что ассистент должен ПРЕКРАТИТЬ общение,\n"
        "   ЗАВЕРШИТЬ диалог, не продолжать переписку и т.п. —\n"
        "   то ассистент НЕ должен задавать новых вопросов и не должен предлагать продолжение диалога.\n"
        "   Если ассистент прислал только слово 'END', нужно оценивать как 1, так как таким образом он явно завершил диалог.\n"
        "   Любые дополнительные вопросы, предложения вернуться к общению и т.п. = нарушение → score: 0.\n"
        "\n"
        "2) Если в expected_behavior написано, что ассистент ДОЛЖЕН завершить сообщение маркером END,\n"
        "   то в ответе ассистента ОБЯЗАТЕЛЬНО должна присутствовать подстрока 'END' (заглавными латинскими буквами).\n"
        "   - Если END нет в ответе ассистента — ставь score: 0.\n"
        "\n"
        "3) Если expected_behavior описывает конкретную структуру реакции (например:\n"
        "   «вежливо поблагодарить, извиниться и больше не писать»),\n"
        "   то ответ должен: по смыслу соответствовать (есть благодарность/извинение/отказ),\n"
        "4) Если в expected_behavior нет явного требования по END и нет жёстких формулировок,\n"
        "   всё равно проверяй по смыслу: ответ должен логически соответствовать описанию.\n"
        "\n"
        "5) Не смягчай критерии: даже если ответ вежливый и выглядит разумно, но не следует\n"
        "   буквальным требованиям expected_behavior (например, продолжает диалог вместо остановки),\n"
        "   ты ОБЯЗАН поставить score: 0.\n"
        "\n"
        "6) Не добавляй никакого текста вне JSON.\n"
    )

    payload = eval_instruction + "\n\n" + json.dumps(
        {
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "expected_behavior": scenario.expected_behavior,
            "step": step,
            "candidate_message": candidate_message,
            "assistant_reply": assistant_reply,
        },
        ensure_ascii=False,
    )

    response = client.responses.create(
        model=EVAL_MODEL,
        input=payload,
    )
    text = (getattr(response, "output_text", "") or "").strip()

    try:
        data = json.loads(text)
        score = int(data.get("score", 0))
        if score not in (0, 1):
            score = 0
        comment = str(data.get("comment", "")).strip() or "No comment."
    except Exception:
        score = 0
        comment = f"Failed to parse eval model output: {text[:200]}"

    return score, comment


# -----------------------
# Основной раннер
# -----------------------

def run_scenarios(
    csv_path: pathlib.Path,
    messages_per_scenario: int,
    max_scenarios: int | None = None,
) -> pathlib.Path:
    ensure_dirs()

    print(f"[init] Loading scenarios from CSV: {csv_path}")
    scenarios = load_scenarios(csv_path)
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    if not scenarios:
        raise ValueError("No scenarios loaded from CSV - nothing to run.")

    print(f"[init] Total scenarios loaded: {len(scenarios)}")

    if not CFG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CFG_PATH}")
    cfg = load_yaml(CFG_PATH)
    print(f"[init] Loaded config: {CFG_PATH}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    started_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    total_score = 0
    messages_total = 0

    # учёт токенов по компонентам
    usage = {
        "candidate_generator": _blank_usage(),
        "screening_assistant": _blank_usage(),
        "evaluator": _blank_usage(),
    }

    scenarios_payload: List[Dict[str, Any]] = []
    failed_messages: List[Dict[str, Any]] = []

    print(
        f"[run] Starting screening_scenarios run_id={run_id} | "
        f"scenarios={len(scenarios)} | messages_per_scenario={messages_per_scenario}"
    )

    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n[scenario {idx}/{len(scenarios)}] #{scenario.index}: {scenario.name}")
        print("  - generating candidate messages...")

        # 1) генерируем N реплик кандидата
        gen_response = client.responses.create(
            model=GEN_MODEL,
            input="ping",  # small no-op to ensure client is ok (optional)
        )
        _accumulate_usage(usage["candidate_generator"], getattr(gen_response, "usage", None))

        candidate_messages = generate_candidate_messages_for_scenario(
            client, scenario, messages_per_scenario
        )

        print(f"  - generated {len(candidate_messages)} candidate messages:")
        for i, msg in enumerate(candidate_messages, start=1):
            preview = msg.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"      [{i}] {preview}")

        # 2) прогоняем через screening_assistant
        print("  - running screening_assistant for scenario...")
        turns = run_screening_assistant_turns(
            cfg, scenario, candidate_messages, usage["screening_assistant"]
        )
        print(f"  - got {len(turns)} assistant replies")

        # 3) оцениваем каждый turn
        scenario_score = 0
        print("  - evaluating turns against expected_behavior...")
        for turn in turns:
            step = turn["step"]
            cand_msg = turn["candidate_message"]
            reply = turn["assistant_reply"]

            # отдельный вызов для учёта usage ревьюера (no-op ping)
            eval_response = client.responses.create(
                model=EVAL_MODEL,
                input="ping",  # no-op
            )
            _accumulate_usage(usage["evaluator"], getattr(eval_response, "usage", None))

            score, comment = evaluate_turn(
                client=client,
                scenario=scenario,
                step=step,
                candidate_message=cand_msg,
                assistant_reply=reply,
            )
            turn["score"] = score
            turn["comment"] = comment

            messages_total += 1
            total_score += score
            scenario_score += score

            status = "OK" if score == 1 else "FAIL"
            reply_preview = reply.replace("\n", " ")
            if len(reply_preview) > 100:
                reply_preview = reply_preview[:97] + "..."
            print(f"      step {step}: {status} (score={score}) | reply: {reply_preview}")

            if score == 0:
                failed_messages.append(
                    {
                        "scenario_index": scenario.index,
                        "scenario_name": scenario.name,
                        "step": step,
                        "candidate_message": cand_msg,
                        "assistant_reply": reply,
                        "comment": comment,
                    }
                )

        scenarios_payload.append(
            {
                "scenario_index": scenario.index,
                "scenario_name": scenario.name,
                "steps_run": len(turns),
                "total_score": scenario_score,
                "passed": scenario_score == len(turns),
                "turns": turns,
            }
        )

        print(
            f"  - scenario result: score={scenario_score}/{len(turns)} | "
            f"passed={'YES' if scenario_score == len(turns) else 'NO'}"
        )

    score_rate = (total_score / messages_total) if messages_total else 0.0

    # достаём конфиг screening_assistant, чтобы записать prompt_id и prompt_version в отчёт
    sa_cfg = _component_cfg(cfg, "screening_assistant")

    print("\n[summary] All scenarios processed.")
    print(
        f"[summary] messages_total={messages_total}, "
        f"score_total={total_score}, score_rate={score_rate:.3f}"
    )

    report: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "csv_path": str(csv_path),
        "scenarios_total": len(scenarios),
        "messages_total": messages_total,
        "score_total": total_score,
        "score_rate": score_rate,
        "scenarios": scenarios_payload,
        "failed_messages": failed_messages,
        "token_usage": usage,
        "models": {
            "candidate_generator": GEN_MODEL,
            "screening_assistant": {
                "prompt_id": sa_cfg.get("prompt_id"),
                "prompt_version": sa_cfg.get("prompt_version"),
            },
            "evaluator": EVAL_MODEL,
        },
    }

    out_path = REPORTS_DIR / f"screening_scenarios_report_{run_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] Screening scenarios report saved to: {out_path}")

    return out_path



# -----------------------
# CLI
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral screening scenarios against screening_assistant."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to CSV with scenarios (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--messages-per-scenario",
        type=int,
        default=DEFAULT_MESSAGES_PER_SCENARIO,
        help=f"How many messages to test per scenario (default: {DEFAULT_MESSAGES_PER_SCENARIO})",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit number of scenarios (for quick debug). Default: all.",
    )

    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    report_path = run_scenarios(
        csv_path=csv_path,
        messages_per_scenario=args.messages_per_scenario,
        max_scenarios=args.max_scenarios,
    )
    print("Screening scenarios report ->", report_path)


if __name__ == "__main__":
    main()

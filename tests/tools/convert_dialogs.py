from __future__ import annotations

import argparse
import json
import pathlib
import re

LINK_RE = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"(?:(?:\+?\d)[\d\-\s\(\)]{6,}\d)")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.replace("\r", "").strip()
    text = LINK_RE.sub("<LINK>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    return text


def convert_raw_to_parsed(
    raw_path: pathlib.Path,
    parsed_path: pathlib.Path,
) -> None:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    parsed_lines: list[str] = []
    for message in data:
        text = normalize_text(message.get("text"))
        if not text:
            continue
        role = "assistant" if message.get("is_owner") else "candidate"
        parsed_lines.append(json.dumps({"role": role, "text": text}, ensure_ascii=False))
    parsed_path.write_text("\n".join(parsed_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="tests/fixtures/dialogs_raw")
    parser.add_argument("--out_parsed_dir", default="tests/fixtures/dialogs_parsed")
    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_parsed = pathlib.Path(args.out_parsed_dir)
    out_parsed.mkdir(parents=True, exist_ok=True)

    for raw_file in in_dir.glob("*.json"):
        parsed_file = out_parsed / (raw_file.stem.replace("messages_", "") + ".dialog.jsonl")
        convert_raw_to_parsed(raw_file, parsed_file)
        print(f"OK {raw_file.name} -> {parsed_file.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-command batch runner for nlayers experiments.

This script wraps `nlayers_interface.py` and provides predefined cases,
so you can launch multiple nlayers batches with a single command.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


PRESET_CASES: dict[str, dict[str, list[int]]] = {
    "quick": {
        "dpa": [1],
        "les": [3],
        "sog": [6],
    },
    "compare": {
        "dpa": [1, 3, 6],
        "les": [1, 3, 6],
        "sog": [1, 3, 6],
    },
    "sog_scan": {
        "sog": [1, 2, 3, 4, 5, 6],
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predefined nlayers experiment cases for NaCl with one command."
    )
    parser.add_argument(
        "--case",
        choices=sorted(PRESET_CASES.keys()),
        default="compare",
        help="Preset nlayers case.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["dpa", "les", "sog"],
        default=None,
        help="Optional subset of models, e.g. --models sog les.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run training jobs. Without this flag, only generate input JSONs.",
    )
    parser.add_argument(
        "--numb-steps",
        type=int,
        default=None,
        help="Optional override for training.numb_steps.",
    )
    parser.add_argument(
        "--dp-cmd",
        default="dp --pt train",
        help="Training command prefix when --run is set.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining models if one model batch fails.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("batch_summaries"),
        help="Directory to write per-model and merged summaries.",
    )
    return parser.parse_args()


def _build_cmd(
    interface_script: Path,
    model: str,
    nlayers_list: list[int],
    summary_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd: list[str] = [
        "python",
        str(interface_script),
        "--model",
        model,
        "--nlayers-list",
        *[str(x) for x in nlayers_list],
        "--summary-file",
        str(summary_file),
    ]
    if args.numb_steps is not None:
        cmd += ["--numb-steps", str(args.numb_steps)]
    if args.run:
        cmd += ["--run", "--dp-cmd", args.dp_cmd]
    if args.continue_on_error:
        cmd += ["--continue-on-error"]
    return cmd


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    args = _parse_args()
    base_dir = Path(__file__).resolve().parent
    interface_script = base_dir / "nlayers_interface.py"
    if not interface_script.exists():
        raise FileNotFoundError(f"Missing interface script: {interface_script}")

    case_cfg = PRESET_CASES[args.case]
    selected_models = args.models if args.models is not None else list(case_cfg.keys())

    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else (base_dir / args.summary_dir)
    summary_dir = summary_dir / args.case
    summary_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, Any] = {
        "case": args.case,
        "run": bool(args.run),
        "numb_steps": args.numb_steps,
        "dp_cmd": args.dp_cmd,
        "models": {},
        "failed_models": [],
    }

    for model in selected_models:
        if model not in case_cfg:
            print(f"skip model={model}: not configured in case={args.case}")
            continue
        nlayers_list = case_cfg[model]
        summary_file = summary_dir / f"summary_{model}.json"
        cmd = _build_cmd(interface_script, model, nlayers_list, summary_file, args)

        print("==== Run Model Batch ====")
        print(f"model: {model}")
        print(f"nlayers: {nlayers_list}")
        print("cmd:", " ".join(shlex.quote(c) for c in cmd))

        proc = subprocess.run(cmd, cwd=str(base_dir), check=False)
        model_entry: dict[str, Any] = {
            "nlayers": nlayers_list,
            "summary_file": str(summary_file),
            "return_code": proc.returncode,
            "status": "ok" if proc.returncode == 0 else "failed",
        }
        if summary_file.exists():
            model_entry["results"] = _load_json(summary_file)
        else:
            model_entry["results"] = []

        merged["models"][model] = model_entry

        if proc.returncode != 0:
            merged["failed_models"].append(model)
            if not args.continue_on_error:
                break

    merged_file = summary_dir / "summary_all_models.json"
    merged_file.write_text(json.dumps(merged, indent=2) + "\n")

    print("==== Case Summary ====")
    print(f"case: {args.case}")
    print(f"summary_dir: {summary_dir}")
    print(f"merged_summary: {merged_file}")
    print(f"failed_models: {merged['failed_models']}")

    if len(merged["failed_models"]) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

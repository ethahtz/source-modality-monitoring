"""
GPT LLM-as-judge over **arbitrary-label** behavioral JSON
(``prompt_evaluation_arbitrary_labels.py`` / ``behavioral_evaluation_arbitrary_labels``).

Same pipeline as ``gpt_judge_evaluation.py``, but prompts and JSON schemas refer to
``label_1`` / ``label_2`` (mapped to image / caption content) instead of the words
``image`` / ``text``, matching how the VLM is instructed (e.g. "Report the information
associated with label Dax.").

Input layout::
    {work_dir}/results/behavioral_evaluation_arbitrary_labels/{model_safe}/{dataset}/
        {input_type}/{modality}/labels_{label_1}_{label_2}/arbitrary_{order}_s{seed}.json

Default GPT output mirrors the input path with ``results/`` → ``results_gpt_eval/``.

Requires the same environment and packages as ``gpt_judge_evaluation.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import gpt_judge_evaluation as gje
from gpt_judge_evaluation import (
    OPENROUTER_DEFAULT_BASE_URL,
    PINNED_DEFAULT_MODEL,
    _format_caption_block,
    _gpt_label_to_outcome,
    _load_dotenv,
    _optional_openrouter_headers,
    _resolve_api_key_for_request,
    _strip_key,
)

# Populated from source JSON ``args`` before each run (``run_sync`` loads JSON internally;
# ``main`` pre-loads once to set these before calling into gje).
_ARB_LABEL_1: str = "Dax"
_ARB_LABEL_2: str = "Wug"

# PINNED_DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"


def _set_arb_labels_from_run_args(run_args: Dict[str, Any]) -> None:
    global _ARB_LABEL_1, _ARB_LABEL_2
    _ARB_LABEL_1 = str(run_args.get("label_1", "Dax"))
    _ARB_LABEL_2 = str(run_args.get("label_2", "Wug"))


def _judge_json_schema_response_format_arb(input_type: str) -> Dict[str, Any]:
    """Structured output using label_1 / label_2 enums (OpenAI strict json_schema)."""
    if input_type == "inconsistent":
        schema = {
            "type": "object",
            "properties": {
                "closer_to": {"type": "string", "enum": ["label_1", "label_2", "neither"]},
            },
            "required": ["closer_to"],
            "additionalProperties": False,
        }
        name = "JudgeArbInconsistent"
    elif input_type == "image_only":
        schema = {
            "type": "object",
            "properties": {"aligns_with_label_1": {"type": "boolean"}},
            "required": ["aligns_with_label_1"],
            "additionalProperties": False,
        }
        name = "JudgeArbImageOnly"
    elif input_type == "text_only":
        schema = {
            "type": "object",
            "properties": {"aligns_with_label_2": {"type": "boolean"}},
            "required": ["aligns_with_label_2"],
            "additionalProperties": False,
        }
        name = "JudgeArbTextOnly"
    elif input_type == "consistent":
        schema = {
            "type": "object",
            "properties": {"aligns_with_target": {"type": "boolean"}},
            "required": ["aligns_with_target"],
            "additionalProperties": False,
        }
        name = "JudgeArbConsistent"
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _response_format_arb(structured: str, input_type: str) -> Dict[str, Any]:
    if structured == "json_schema":
        return _judge_json_schema_response_format_arb(input_type)
    return {"type": "json_object"}


def _build_messages_arb(
    input_type: str,
    modality_to_report: str,
    prediction: str,
    image_captions: List[str],
    paired_caption: Optional[str],
) -> List[Dict[str, str]]:
    """
    Judge prompts aligned with arbitrary-label VLM instructions: label_1 ↔ image stream,
    label_2 ↔ written caption (see ``get_arbitrary_label_prompt_template_args``).
    """
    label_1 = _ARB_LABEL_1
    label_2 = _ARB_LABEL_2
    cap_block = _format_caption_block(image_captions)
    pred = prediction.strip() or "(empty)"
    paired = paired_caption if paired_caption is not None else "(none)"

    sys = (
        "You are a careful evaluator for vision-language model outputs. "
        "Use only the user message. Reply with JSON only, matching the requested shape."
    )

    if input_type == "inconsistent":
        user = f"""The model saw two sources of information marked with arbitrary labels **{label_1}** and **{label_2}**.
Content associated with **{label_1}** comes from the *image*; content associated with **{label_2}** is the *written caption*.
Those two sources describe *different* scenes (inconsistent). The model was instructed to report information for one of the labels (as in the original task).

Reference A — content associated with label **{label_1}** (image; human reference captions):
{cap_block}

Reference B — content associated with label **{label_2}** (written caption next to the image; which disagrees with the image):
{paired}

Model answer:
\"\"\"{pred}\"\"\"

Which reference is the answer primarily grounded in? Output only the JSON object with closer_to one of: label_1, label_2, neither."""

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _arb_parsed_to_standard(input_type: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Map label_1/label_2 judge fields to the schema expected by ``_gpt_label_to_outcome`` (image/text)."""
    if input_type == "inconsistent":
        raw = str(parsed.get("closer_to", "")).lower().strip()
        if raw == "label_1":
            return {"closer_to": "image"}
        if raw == "label_2":
            return {"closer_to": "text"}
        if raw == "neither":
            return {"closer_to": "neither"}
        if raw in ("image", "text", "neither"):
            return {"closer_to": raw}
        raise ValueError(f"Invalid closer_to: {raw!r}")
    if input_type == "image_only":
        v = parsed.get("aligns_with_label_1")
        if v is None and "aligns_with_image" in parsed:
            v = parsed["aligns_with_image"]
        if v is None:
            raise ValueError("Missing aligns_with_label_1")
        return {"aligns_with_image": bool(v)}
    if input_type == "text_only":
        v = parsed.get("aligns_with_label_2")
        if v is None and "aligns_with_text" in parsed:
            v = parsed["aligns_with_text"]
        if v is None:
            raise ValueError("Missing aligns_with_label_2")
        return {"aligns_with_text": bool(v)}
    if input_type == "consistent":
        if "aligns_with_target" not in parsed:
            raise ValueError("Missing aligns_with_target")
        return {"aligns_with_target": bool(parsed["aligns_with_target"])}
    raise ValueError(f"Unknown input_type: {input_type}")


def _gpt_label_to_outcome_arb(
    input_type: str,
    modality_to_report: str,
    parsed: Dict[str, Any],
) -> Tuple[str, str]:
    std = _arb_parsed_to_standard(input_type, parsed)
    return _gpt_label_to_outcome(input_type, modality_to_report, std)


# --- Path helpers (model_name uses "/" in JSON; on-disk dirs use "_") ---

def _model_dir_on_disk(model_name: str) -> str:
    return model_name.replace("/", "_")


def _arbitrary_behavioral_json_path(
    work_dir: str,
    model_name: str,
    dataset: str,
    input_type: str,
    modality: str,
    label_1: str,
    label_2: str,
    order: str,
    seed: int,
) -> str:
    fname = f"arbitrary_{order}_s{seed}.json"
    return os.path.join(
        work_dir,
        "results",
        "behavioral_evaluation_arbitrary_labels",
        _model_dir_on_disk(model_name),
        dataset,
        input_type,
        modality,
        f"labels_{label_1}_{label_2}",
        fname,
    )


def _gpt_arb_output_path(
    work_dir: str,
    model_name: str,
    dataset: str,
    input_type: str,
    modality: str,
    label_1: str,
    label_2: str,
    order: str,
    seed: int,
) -> str:
    fname = f"arbitrary_{order}_s{seed}.json"
    return os.path.join(
        work_dir,
        "results_gpt_eval",
        "behavioral_evaluation_arbitrary_labels",
        _model_dir_on_disk(model_name),
        dataset,
        input_type,
        modality,
        f"labels_{label_1}_{label_2}",
        fname,
    )


def _gpt_arb_output_path_from_input_json(input_path: str) -> Optional[str]:
    abs_in = os.path.normpath(os.path.abspath(input_path))
    sep = os.sep
    marker = f"{sep}results{sep}"
    if marker not in abs_in:
        return None
    return abs_in.replace(marker, f"{sep}results_gpt_eval{sep}", 1)


@contextmanager
def _patched_gje():
    """Temporarily swap behavioral judge helpers for arbitrary-label versions."""
    old = {
        "_build_messages": gje._build_messages,
        "_judge_json_schema_response_format": gje._judge_json_schema_response_format,
        "_response_format": gje._response_format,
        "_gpt_label_to_outcome": gje._gpt_label_to_outcome,
    }
    gje._build_messages = _build_messages_arb
    gje._judge_json_schema_response_format = _judge_json_schema_response_format_arb
    gje._response_format = _response_format_arb
    gje._gpt_label_to_outcome = _gpt_label_to_outcome_arb
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(gje, k, v)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT judge over arbitrary-label behavioral JSON (behavioral_evaluation_arbitrary_labels)."
    )
    p.add_argument("--work_dir", type=str, default=os.environ.get("WORKDIR", "."))
    p.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Full path to source arbitrary-label results JSON. If set, path flags are ignored. "
        "Default output: same path with results/ → results_gpt_eval/ (unless --output_json).",
    )
    p.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    p.add_argument("--dataset", type=str, default="flickr30k")
    p.add_argument(
        "--input_type",
        type=str,
        default="inconsistent",
        choices=["consistent", "inconsistent", "text_only", "image_only"],
    )
    p.add_argument(
        "--modality_to_report",
        type=str,
        default="image",
        choices=["image", "text"],
    )
    p.add_argument("--order", type=str, default="icq")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_1", type=str, default="Dax", help="Must match the source JSON labels_* dir.")
    p.add_argument("--label_2", type=str, default="Wug")
    p.add_argument("--gpt_model", type=str, default=PINNED_DEFAULT_MODEL)
    p.add_argument("--mode", type=str, default="async", choices=["async", "sync", "batch"])
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--structured_output", type=str, default="json_schema", choices=["json_schema", "json_object"])
    p.add_argument("--max_completion_tokens", type=int, default=128)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--legacy_json_parse", action="store_true")
    p.add_argument("--debug_http", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--batch_poll_seconds", type=float, default=15.0)
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--env_file", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--openrouter", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    _load_dotenv(args.env_file, args.work_dir)

    base_url: Optional[str] = args.base_url
    if args.openrouter:
        base_url = base_url or OPENROUTER_DEFAULT_BASE_URL

    api_key = _resolve_api_key_for_request(base_url, args.openrouter)
    if not api_key:
        wd = os.path.abspath(args.work_dir)
        raise RuntimeError(
            "No API key: set OPENAI_API_KEY or OPENROUTER_API_KEY, "
            f"or add to .env (see gpt_judge_evaluation.py). work_dir={wd}"
        )

    default_headers = _optional_openrouter_headers()

    if args.mode == "batch" and base_url and "openrouter.ai" in base_url:
        raise RuntimeError("OpenAI Batch API is not available on OpenRouter; use --mode async or sync.")

    if args.input_json:
        input_path = os.path.abspath(args.input_json)
    else:
        input_path = _arbitrary_behavioral_json_path(
            work_dir=os.path.abspath(args.work_dir),
            model_name=args.model_name,
            dataset=args.dataset,
            input_type=args.input_type,
            modality=args.modality_to_report,
            label_1=args.label_1,
            label_2=args.label_2,
            order=args.order,
            seed=args.seed,
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing input JSON: {input_path}")

    with open(input_path, "r") as f:
        preload = json.load(f)
    run_args_pre = preload.get("args") or {}
    _set_arb_labels_from_run_args(run_args_pre)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
    elif args.input_json:
        derived = _gpt_arb_output_path_from_input_json(input_path)
        if derived is not None:
            output_path = derived
        else:
            output_path = _gpt_arb_output_path(
                work_dir=os.path.abspath(args.work_dir),
                model_name=args.model_name,
                dataset=args.dataset,
                input_type=args.input_type,
                modality=args.modality_to_report,
                label_1=args.label_1,
                label_2=args.label_2,
                order=args.order,
                seed=args.seed,
            )
    else:
        output_path = _gpt_arb_output_path(
            work_dir=os.path.abspath(args.work_dir),
            model_name=args.model_name,
            dataset=args.dataset,
            input_type=args.input_type,
            modality=args.modality_to_report,
            label_1=args.label_1,
            label_2=args.label_2,
            order=args.order,
            seed=args.seed,
        )

    online = dict(
        input_path=input_path,
        output_path=output_path,
        gpt_model=args.gpt_model,
        max_instances=args.max_instances,
        start=args.start,
        end=args.end,
        structured=args.structured_output,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        legacy_regex=args.legacy_json_parse,
        debug_http=args.debug_http,
        verbose=args.verbose,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )

    with _patched_gje():
        if args.mode == "async":
            result = asyncio.run(gje.run_async(**online, concurrency=args.concurrency))
        elif args.mode == "sync":
            result = gje.run_sync(**online)
        else:
            result = gje.run_batch(
                input_path=input_path,
                output_path=output_path,
                gpt_model=args.gpt_model,
                max_instances=args.max_instances,
                start=args.start,
                end=args.end,
                structured=args.structured_output,
                max_completion_tokens=args.max_completion_tokens,
                legacy_regex=args.legacy_json_parse,
                poll_seconds=args.batch_poll_seconds,
                verbose=args.verbose,
                api_key=api_key,
                base_url=base_url,
                default_headers=default_headers,
            )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    _print_keys = {"run_args", "instances"}
    print(json.dumps({k: v for k, v in result.items() if k not in _print_keys}, indent=2))
    if "instances" in result:
        print(f"(Omitted {len(result['instances'])} instance rows from stdout; full data is in the output JSON.)")
    print(f"Saved aggregate GPT evaluation to {output_path}")


if __name__ == "__main__":
    main()

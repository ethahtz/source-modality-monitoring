#!/usr/bin/env python3
"""
GPT judge for eval_transformation_vec.py outputs (eval_results.json).

For each row in ``responses``, runs the same inconsistent-task prompt as
``gpt_judge_evaluation.py`` (_build_messages with input_type=inconsistent) on four fields:
  - pred_baseline_image_target_prompt  → modality_to_report = image
  - pred_baseline_text_target_prompt   → modality_to_report = text
  - pred_intervention_image_target_prompt → image
  - pred_intervention_text_target_prompt  → text

Requires OPENAI_API_KEY or OPENROUTER_API_KEY (see gpt_judge_evaluation.py). Prefer
``--mode async`` for OpenRouter (sync path matches gpt_judge and may pass OpenAI-only kwargs).

Example:
  python src/gpt_judge_eval_transformation_vec.py \\
    --input_json results/eval_transformation_vec/.../eval_results.json \\
    --work_dir .
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from gpt_judge_evaluation import (
    OPENROUTER_DEFAULT_BASE_URL,
    PINNED_DEFAULT_MODEL,
    _OPENAI_IMPORT_ERROR,
    _accumulate_counts,
    _build_messages as gpt_build_messages,
    _call_gpt_async,
    _call_gpt_sync,
    _gpt_label_to_outcome,
    _load_dotenv,
    _make_openai_async,
    _make_openai_sync,
    _optional_openrouter_headers,
    _resolve_api_key_for_request,
    _response_format,
)

# (json field name, modality_to_report for outcome mapping)
PREDICTION_SLOTS: List[Tuple[str, str]] = [
    ("pred_baseline_image_target_prompt", "image"),
    ("pred_baseline_text_target_prompt", "text"),
    ("pred_intervention_image_target_prompt", "image"),
    ("pred_intervention_text_target_prompt", "text"),
]

INPUT_TYPE = "inconsistent"


def _default_output_path(input_path: str) -> str:
    """Mirror .../results/... under .../results_gpt_eval/... keeping suffix eval_results.json."""
    abs_in = os.path.normpath(os.path.abspath(input_path))
    sep = os.sep
    marker = f"{sep}results{sep}"
    if marker not in abs_in:
        return abs_in + ".gpt_judged.json"
    out = abs_in.replace(marker, f"{sep}results_gpt_eval{sep}", 1)
    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)
    return out


def _empty_slot_result(
    gpt_error: Optional[str],
) -> Dict[str, Any]:
    return {
        "gpt_parsed": None,
        "gpt_raw_response": None,
        "gpt_completion_id": None,
        "gpt_error": gpt_error,
        "judgment": "neither",
        "outcome": "neither",
    }


def _eval_one_slot_sync(
    client: Any,
    *,
    gpt_model: str,
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    prediction: str,
    image_captions: List[str],
    paired_caption: Optional[str],
    modality_to_report: str,
    debug_http: bool,
    debug_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = gpt_build_messages(
        input_type=INPUT_TYPE,
        modality_to_report=modality_to_report,
        prediction=str(prediction),
        image_captions=image_captions,
        paired_caption=paired_caption,
    )
    try:
        parsed, req_id, raw = _call_gpt_sync(
            client,
            gpt_model,
            messages,
            response_format,
            max_completion_tokens,
            max_retries,
            legacy_regex,
            debug_http,
            debug_state,
        )
        judgment, outcome = _gpt_label_to_outcome(INPUT_TYPE, modality_to_report, parsed)
        return {
            "gpt_parsed": parsed,
            "gpt_raw_response": raw,
            "gpt_completion_id": req_id,
            "gpt_error": None,
            "judgment": judgment,
            "outcome": outcome,
        }
    except Exception as e:  # noqa: BLE001
        out = _empty_slot_result(str(e))
        out["gpt_parsed"] = None
        return out


async def _eval_one_slot_async(
    client: Any,
    semaphore: asyncio.Semaphore,
    *,
    gpt_model: str,
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    prediction: str,
    image_captions: List[str],
    paired_caption: Optional[str],
    modality_to_report: str,
    debug_http: bool,
    debug_lock: asyncio.Lock,
    debug_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = gpt_build_messages(
        input_type=INPUT_TYPE,
        modality_to_report=modality_to_report,
        prediction=str(prediction),
        image_captions=image_captions,
        paired_caption=paired_caption,
    )
    try:
        parsed, req_id, raw = await _call_gpt_async(
            client,
            gpt_model,
            messages,
            response_format,
            max_completion_tokens,
            max_retries,
            legacy_regex,
            debug_http,
            debug_lock,
            debug_state,
            semaphore=semaphore,
        )
        judgment, outcome = _gpt_label_to_outcome(INPUT_TYPE, modality_to_report, parsed)
        return {
            "gpt_parsed": parsed,
            "gpt_raw_response": raw,
            "gpt_completion_id": req_id,
            "gpt_error": None,
            "judgment": judgment,
            "outcome": outcome,
        }
    except Exception as e:  # noqa: BLE001
        return _empty_slot_result(str(e))


def run_sync_all(
    input_path: str,
    output_path: str,
    *,
    gpt_model: str,
    structured: str,
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    api_key: str,
    base_url: Optional[str],
    default_headers: Optional[Dict[str, str]],
    max_responses: Optional[int],
    start: int,
    end: Optional[int],
) -> Dict[str, Any]:
    if _OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("Install openai>=1.0.0") from _OPENAI_IMPORT_ERROR

    with open(input_path) as f:
        data = json.load(f)

    responses: List[Dict[str, Any]] = list(data.get("responses") or [])
    slice_end = len(responses) if end is None else min(end, len(responses))
    slice_start = max(0, start)
    sliced = responses[slice_start:slice_end]
    if max_responses is not None:
        sliced = sliced[:max_responses]

    response_format = _response_format(structured, INPUT_TYPE)
    client = _make_openai_sync(api_key, base_url=base_url, default_headers=default_headers)
    debug_state: Optional[Dict[str, Any]] = (
        {"last_rate_limit_headers": {}, "sample_request_ids": []} if debug_http else None
    )

    aggregate_by_slot: Dict[str, Dict[str, int]] = {
        field: {"correct": 0, "misled": 0, "neither": 0} for field, _ in PREDICTION_SLOTS
    }
    judgment_by_slot: Dict[str, Dict[str, int]] = {
        field: {"image": 0, "text": 0, "neither": 0} for field, _ in PREDICTION_SLOTS
    }
    n_parse_fail = 0
    rows_out: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for offset, row in enumerate(tqdm(sliced, desc="GPT judge transformation-vec (sync)")):
        idx = slice_start + offset
        caps = list(row.get("image_caption_reference") or [])
        paired = row.get("paired_caption")
        judges: Dict[str, Any] = {}

        for field, modality in PREDICTION_SLOTS:
            pred = row.get(field, "")
            if pred is None:
                pred = ""
            slot_res = _eval_one_slot_sync(
                client,
                gpt_model=gpt_model,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                max_retries=max_retries,
                legacy_regex=legacy_regex,
                prediction=str(pred),
                image_captions=caps,
                paired_caption=paired,
                modality_to_report=modality,
                debug_http=debug_http,
                debug_state=debug_state,
            )
            judges[field] = slot_res
            if slot_res.get("gpt_error"):
                n_parse_fail += 1
            o = slot_res.get("outcome", "neither")
            j = slot_res.get("judgment", "neither")
            _accumulate_counts(aggregate_by_slot[field], o)
            if j in judgment_by_slot[field]:
                judgment_by_slot[field][j] += 1

        rows_out.append(
            {
                "response_index": idx,
                "order": row.get("order"),
                "dataset_index": row.get("dataset_index"),
                "gpt_judges": judges,
            }
        )

    elapsed = time.perf_counter() - t0
    meta_keys = (
        "intervention",
        "train_orders_for_checkpoint",
        "train_seed_for_checkpoint",
        "deltas_path",
        "deltas_marker_path",
        "deltas_content_path",
        "eval_orders",
        "split",
        "mscoco_split",
        "layer_idx",
        "layer_depth",
        "num_eval_pairs_built",
        "num_responses_saved",
        "prompt_format",
        "note",
    )
    source_meta = {k: data[k] for k in meta_keys if k in data}

    out: Dict[str, Any] = {
        "source_result_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "eval_mode": "sync",
        "input_type": INPUT_TYPE,
        "gpt_model": gpt_model,
        "structured_output": structured,
        "prediction_slots": [list(t) for t in PREDICTION_SLOTS],
        "api_base_url": base_url,
        "source_meta": source_meta,
        "eval_slice": {"start": slice_start, "end": slice_start + len(sliced), "n": len(sliced)},
        "aggregate_by_slot": aggregate_by_slot,
        "judgment_hist_by_slot": judgment_by_slot,
        "n_slot_errors": n_parse_fail,
        "wall_time_seconds": round(elapsed, 3),
        "responses": rows_out,
    }
    if debug_state:
        out["debug_http"] = {
            "sample_request_ids": debug_state.get("sample_request_ids", []),
            "last_rate_limit_headers": debug_state.get("last_rate_limit_headers", {}),
        }
    return out


async def run_async_all(
    input_path: str,
    output_path: str,
    *,
    gpt_model: str,
    structured: str,
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    concurrency: int,
    api_key: str,
    base_url: Optional[str],
    default_headers: Optional[Dict[str, str]],
    max_responses: Optional[int],
    start: int,
    end: Optional[int],
) -> Dict[str, Any]:
    if _OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("Install openai>=1.0.0") from _OPENAI_IMPORT_ERROR

    with open(input_path) as f:
        data = json.load(f)

    responses: List[Dict[str, Any]] = list(data.get("responses") or [])
    slice_end = len(responses) if end is None else min(end, len(responses))
    slice_start = max(0, start)
    sliced = responses[slice_start:slice_end]
    if max_responses is not None:
        sliced = sliced[:max_responses]

    response_format = _response_format(structured, INPUT_TYPE)
    client = _make_openai_async(api_key, base_url=base_url, default_headers=default_headers)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    debug_lock = asyncio.Lock()
    debug_state: Optional[Dict[str, Any]] = (
        {"last_rate_limit_headers": {}, "sample_request_ids": []} if debug_http else None
    )

    tasks = []
    task_meta: List[Tuple[int, str, str]] = []

    for offset, row in enumerate(sliced):
        resp_idx = slice_start + offset
        caps = list(row.get("image_caption_reference") or [])
        paired = row.get("paired_caption")
        for field, modality in PREDICTION_SLOTS:
            pred = row.get(field, "") or ""
            task_meta.append((resp_idx, field, modality))
            tasks.append(
                _eval_one_slot_async(
                    client,
                    semaphore,
                    gpt_model=gpt_model,
                    response_format=response_format,
                    max_completion_tokens=max_completion_tokens,
                    max_retries=max_retries,
                    legacy_regex=legacy_regex,
                    prediction=str(pred),
                    image_captions=caps,
                    paired_caption=paired,
                    modality_to_report=modality,
                    debug_http=debug_http,
                    debug_lock=debug_lock,
                    debug_state=debug_state,
                )
            )

    t0 = time.perf_counter()
    results = await tqdm_asyncio.gather(*tasks, desc="GPT judge transformation-vec (async)")
    elapsed = time.perf_counter() - t0

    aggregate_by_slot: Dict[str, Dict[str, int]] = {
        field: {"correct": 0, "misled": 0, "neither": 0} for field, _ in PREDICTION_SLOTS
    }
    judgment_by_slot: Dict[str, Dict[str, int]] = {
        field: {"image": 0, "text": 0, "neither": 0} for field, _ in PREDICTION_SLOTS
    }
    n_slot_errors = 0

    by_response_idx: Dict[int, Dict[str, Any]] = {}
    for (resp_idx, field, _mod), slot_res in zip(task_meta, results):
        if resp_idx not in by_response_idx:
            by_response_idx[resp_idx] = {
                "response_index": resp_idx,
                "order": None,
                "dataset_index": None,
                "gpt_judges": {},
            }
        by_response_idx[resp_idx]["gpt_judges"][field] = slot_res
        if slot_res.get("gpt_error"):
            n_slot_errors += 1
        o = slot_res.get("outcome", "neither")
        j = slot_res.get("judgment", "neither")
        _accumulate_counts(aggregate_by_slot[field], o)
        if j in judgment_by_slot[field]:
            judgment_by_slot[field][j] += 1

    for offset, row in enumerate(sliced):
        resp_idx = slice_start + offset
        rec = by_response_idx[resp_idx]
        rec["order"] = row.get("order")
        rec["dataset_index"] = row.get("dataset_index")

    rows_out = [by_response_idx[slice_start + i] for i in range(len(sliced))]

    meta_keys = (
        "intervention",
        "train_orders_for_checkpoint",
        "train_seed_for_checkpoint",
        "deltas_path",
        "deltas_marker_path",
        "deltas_content_path",
        "eval_orders",
        "split",
        "mscoco_split",
        "layer_idx",
        "layer_depth",
        "num_eval_pairs_built",
        "num_responses_saved",
        "prompt_format",
        "note",
    )
    source_meta = {k: data[k] for k in meta_keys if k in data}

    out: Dict[str, Any] = {
        "source_result_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "eval_mode": "async",
        "concurrency": concurrency,
        "input_type": INPUT_TYPE,
        "gpt_model": gpt_model,
        "structured_output": structured,
        "prediction_slots": [list(t) for t in PREDICTION_SLOTS],
        "api_base_url": base_url,
        "source_meta": source_meta,
        "eval_slice": {"start": slice_start, "end": slice_start + len(sliced), "n": len(sliced)},
        "aggregate_by_slot": aggregate_by_slot,
        "judgment_hist_by_slot": judgment_by_slot,
        "n_slot_errors": n_slot_errors,
        "wall_time_seconds": round(elapsed, 3),
        "responses": rows_out,
    }
    if debug_state:
        out["debug_http"] = {
            "sample_request_ids": debug_state.get("sample_request_ids", []),
            "last_rate_limit_headers": debug_state.get("last_rate_limit_headers", {}),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT judge for eval_transformation_vec JSON outputs.")
    p.add_argument("--input_json", type=str, required=True)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--work_dir", type=str, default=os.environ.get("WORKDIR", "."))
    p.add_argument("--gpt_model", type=str, default=PINNED_DEFAULT_MODEL)
    p.add_argument("--mode", type=str, default="async", choices=["async", "sync"])
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--structured_output",
        type=str,
        default="json_schema",
        choices=["json_schema", "json_object"],
    )
    p.add_argument("--max_completion_tokens", type=int, default=128)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--legacy_json_parse", action="store_true")
    p.add_argument("--debug_http", action="store_true")
    p.add_argument("--max_responses", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--env_file", type=str, default=None)
    p.add_argument("--base_url", type=str, default=None)
    p.add_argument("--openrouter", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv(args.env_file, args.work_dir)

    base_url: Optional[str] = args.base_url
    if args.openrouter:
        base_url = base_url or OPENROUTER_DEFAULT_BASE_URL

    api_key = _resolve_api_key_for_request(base_url, args.openrouter)
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY (see gpt_judge_evaluation.py --help)."
        )

    default_headers = _optional_openrouter_headers()
    input_path = os.path.abspath(args.input_json)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    output_path = os.path.abspath(args.output_json) if args.output_json else _default_output_path(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    common = dict(
        input_path=input_path,
        output_path=output_path,
        gpt_model=args.gpt_model,
        structured=args.structured_output,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        legacy_regex=args.legacy_json_parse,
        debug_http=args.debug_http,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        max_responses=args.max_responses,
        start=args.start,
        end=args.end,
    )

    if args.mode == "async":
        result = asyncio.run(run_async_all(**common, concurrency=args.concurrency))
    else:
        result = run_sync_all(**common)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    slim = {k: v for k, v in result.items() if k != "responses"}
    print(json.dumps(slim, indent=2))
    print(f"(Omitted {len(result['responses'])} response rows from stdout; full data in {output_path})")


if __name__ == "__main__":
    main()

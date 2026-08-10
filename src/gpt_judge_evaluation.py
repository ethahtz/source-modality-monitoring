"""
Re-evaluate saved behavioral evaluation JSON with an OpenAI GPT judge.

Reads results produced by prompt_evaluation.py (instances with model_prediction,
true_label {image_captions, paired_caption}, optional judge_label from BERT).
Writes aggregate metrics and per-instance rows (original instance fields plus
``gpt_judge``). Default output path mirrors the input JSON path with
``results/`` replaced by ``results_gpt_eval/`` when using ``--input_json``; otherwise
``work_dir/results_gpt_eval/behavioral_evaluation/...`` matches the behavioral layout.

Requires: ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY`` (environment or .env), packages
openai>=1.0.0, python-dotenv.

Use **OpenRouter** by setting ``OPENROUTER_API_KEY`` and e.g. ``--openrouter`` or
``--base_url https://openrouter.ai/api/v1`` (see ``--help``). Optional headers:
``OPENROUTER_HTTP_REFERER``, ``OPENROUTER_APP_TITLE``.

Online evaluation defaults to async concurrent requests. Large offline re-evals can use
``--mode batch`` (OpenAI Batch API only; not supported on OpenRouter).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

try:
    from dotenv import load_dotenv as _load_dotenv_impl
except ImportError:  # pragma: no cover
    _load_dotenv_impl = None  # type: ignore

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError as e:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = e
else:
    _OPENAI_IMPORT_ERROR = None

# PINNED_DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"
# PINNED_DEFAULT_MODEL = "openai/gpt-5.4-nano"
PINNED_DEFAULT_MODEL = "openai/gpt-5.4-mini"

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _strip_key(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.strip()
    return s if s else None


def _resolve_api_key_for_request(
    base_url: Optional[str],
    openrouter_flag: bool,
) -> Optional[str]:
    """
    Pick the right key when both OPENAI_API_KEY and OPENROUTER_API_KEY are set:
    OpenRouter endpoints must use OPENROUTER_API_KEY (OpenAI keys are invalid there).
    """
    is_openrouter = openrouter_flag or (
        base_url is not None and "openrouter.ai" in base_url.lower()
    )
    if is_openrouter:
        return _strip_key(os.environ.get("OPENROUTER_API_KEY")) or _strip_key(
            os.environ.get("OPENAI_API_KEY")
        )
    return _strip_key(os.environ.get("OPENAI_API_KEY")) or _strip_key(
        os.environ.get("OPENROUTER_API_KEY")
    )


def _optional_openrouter_headers() -> Optional[Dict[str, str]]:
    """Optional attribution headers for OpenRouter (https://openrouter.ai/docs)."""
    h: Dict[str, str] = {}
    ref = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if ref:
        h["HTTP-Referer"] = ref
    if title:
        h["X-Title"] = title
    return h or None


def _make_openai_sync(
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    return OpenAI(**kwargs)


def _make_openai_async(
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Any:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package not available")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    return AsyncOpenAI(**kwargs)


def _load_dotenv(env_file: Optional[str], work_dir: str) -> None:
    """Populate os.environ from a .env file (does not override existing vars)."""
    if _load_dotenv_impl is None:
        return
    wd = os.path.abspath(work_dir)
    if env_file:
        path = env_file if os.path.isabs(env_file) else os.path.normpath(os.path.join(os.getcwd(), env_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--env_file not found: {path}")
        _load_dotenv_impl(path)
        return
    candidate = os.path.join(wd, ".env")
    if os.path.isfile(candidate):
        _load_dotenv_impl(candidate)
        return
    _load_dotenv_impl()  # find .env upward from cwd


def _behavioral_json_path(
    work_dir: str,
    modify: str,
    model_name: str,
    dataset: str,
    input_type: str,
    modality: str,
    prompt_format: str,
    order: str,
    seed: int,
) -> str:
    fname = f"{prompt_format}_{order}_s{seed}.json"
    return os.path.join(
        work_dir,
        "results",
        "behavioral_evaluation",
        f"modification_{modify}",
        model_name,
        dataset,
        input_type,
        modality,
        fname,
    )


def _gpt_output_path(
    work_dir: str,
    modify: str,
    model_name: str,
    dataset: str,
    input_type: str,
    modality: str,
    prompt_format: str,
    order: str,
    seed: int,
) -> str:
    fname = f"{prompt_format}_{order}_s{seed}.json"
    return os.path.join(
        work_dir,
        "results_gpt_eval",
        "behavioral_evaluation",
        f"modification_{modify}",
        model_name,
        dataset,
        input_type,
        modality,
        fname,
    )


def _gpt_output_path_from_input_json(input_path: str) -> Optional[str]:
    """
    Default output for a behavioral JSON path: same path with ``results/`` replaced by
    ``results_gpt_eval/`` (first occurrence). Returns None if the path has no ``.../results/...`` segment.
    """
    abs_in = os.path.normpath(os.path.abspath(input_path))
    sep = os.sep
    marker = f"{sep}results{sep}"
    if marker not in abs_in:
        return None
    return abs_in.replace(marker, f"{sep}results_gpt_eval{sep}", 1)


def _format_caption_block(image_captions: List[str]) -> str:
    if not image_captions:
        return "(no image captions provided)"
    return "\n".join(f"- {c}" for c in image_captions)


def _judge_json_schema_response_format(input_type: str) -> Dict[str, Any]:
    """Structured output: strict json_schema (matches OpenAI structured outputs rules)."""
    if input_type == "inconsistent":
        schema = {
            "type": "object",
            "properties": {
                "closer_to": {"type": "string", "enum": ["image", "text", "neither"]},
            },
            "required": ["closer_to"],
            "additionalProperties": False,
        }
        name = "JudgeInconsistent"
    elif input_type == "image_only":
        schema = {
            "type": "object",
            "properties": {"aligns_with_image": {"type": "boolean"}},
            "required": ["aligns_with_image"],
            "additionalProperties": False,
        }
        name = "JudgeImageOnly"
    elif input_type == "text_only":
        schema = {
            "type": "object",
            "properties": {"aligns_with_text": {"type": "boolean"}},
            "required": ["aligns_with_text"],
            "additionalProperties": False,
        }
        name = "JudgeTextOnly"
    elif input_type == "consistent":
        schema = {
            "type": "object",
            "properties": {"aligns_with_target": {"type": "boolean"}},
            "required": ["aligns_with_target"],
            "additionalProperties": False,
        }
        name = "JudgeConsistent"
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _response_format(
    structured: str,
    input_type: str,
) -> Dict[str, Any]:
    if structured == "json_schema":
        return _judge_json_schema_response_format(input_type)
    return {"type": "json_object"}


def _parse_content_json(content: str, legacy_regex: bool) -> Dict[str, Any]:
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if not legacy_regex:
            raise
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            raise ValueError(f"No JSON object in model response: {content[:200]!r}") from None
        return json.loads(m.group())


def _build_messages(
    input_type: str,
    modality_to_report: str,
    prediction: str,
    image_captions: List[str],
    paired_caption: Optional[str],
) -> List[Dict[str, str]]:
    cap_block = _format_caption_block(image_captions)
    pred = prediction.strip() or "(empty)"

    sys = (
        "You are a careful evaluator for vision-language model outputs. "
        "Use only the user message. Reply with JSON only, matching the requested shape."
    )

    if input_type == "inconsistent":
        user = f"""The model saw an image and a written caption that describes a *different* scene (inconsistent). It was told to report one modality (image vs text).

Reference A (what is *in the image*):
{cap_block}

Reference B (caption text next to the image; may disagree with the image):
{paired_caption}

Model answer:
\"\"\"{pred}\"\"\"

Which reference is the answer primarily grounded in? Output only the JSON object with closer_to one of: image, text, neither."""

    elif input_type == "image_only":
        user = f"""The model saw only an image. Reference descriptions of the image:
{cap_block}

Model answer:
\"\"\"{pred}\"\"\"

Decide whether the answer is broadly about the same image content. Be permissive: the answer does not need to match the reference exactly.  Output only the JSON object with boolean field aligns_with_image."""
    elif input_type == "text_only":
        user = f"""The model saw only this text:
{paired_caption}

Model answer:
\"\"\"{pred}\"\"\"

Decide whether the answer is broadly about the same text content. Output only the JSON object with boolean field aligns_with_text."""

    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _gpt_label_to_outcome(
    input_type: str,
    modality_to_report: str,
    parsed: Dict[str, Any],
) -> Tuple[str, str]:
    if input_type == "inconsistent":
        raw = str(parsed.get("closer_to", "")).lower().strip()
        if raw not in ("image", "text", "neither"):
            raise ValueError(f"Invalid closer_to: {raw!r}")
        judgment = raw
    elif input_type == "image_only":
        if "aligns_with_image" not in parsed:
            raise ValueError("Missing aligns_with_image")
        judgment = "image" if bool(parsed["aligns_with_image"]) else "neither"
    elif input_type == "text_only":
        if "aligns_with_text" not in parsed:
            raise ValueError("Missing aligns_with_text")
        judgment = "text" if bool(parsed["aligns_with_text"]) else "neither"
    elif input_type == "consistent":
        if "aligns_with_target" not in parsed:
            raise ValueError("Missing aligns_with_target")
        if bool(parsed["aligns_with_target"]):
            judgment = "image" if modality_to_report == "image" else "text"
        else:
            judgment = "neither"
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    if judgment == "neither":
        return judgment, "neither"
    if judgment == "image":
        return judgment, "correct" if modality_to_report == "image" else "misled"
    return judgment, "correct" if modality_to_report == "text" else "misled"


def _accumulate_counts(counts: Dict[str, int], outcome: str) -> None:
    if outcome == "correct":
        counts["correct"] += 1
    elif outcome == "misled":
        counts["misled"] += 1
    elif outcome == "neither":
        counts["neither"] += 1
    else:
        raise ValueError(outcome)


def _instance_eval_record(
    instance_index: int,
    source_inst: Dict[str, Any],
    *,
    gpt_parsed: Optional[Dict[str, Any]],
    gpt_raw_response: Optional[str],
    gpt_completion_id: Optional[str],
    gpt_error: Optional[str],
    judgment: str,
    outcome: str,
) -> Dict[str, Any]:
    """One row: original behavioral fields plus instance_index and gpt_judge block."""
    row = dict(source_inst)
    row["instance_index"] = instance_index
    row["gpt_judge"] = {
        "parsed": gpt_parsed,
        "raw_response": gpt_raw_response,
        "completion_id": gpt_completion_id,
        "error": gpt_error,
        "judgment": judgment,
        "outcome": outcome,
    }
    return row


def _merge_rate_limit_headers(dst: Dict[str, str], headers: Any) -> None:
    if headers is None:
        return
    for key in (
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ):
        v = headers.get(key)
        if v is not None:
            dst[key] = str(v)


def _verbose_log(
    instance_index: int,
    *,
    completion_id: Optional[str],
    api_response_text: str,
    parsed: Optional[Dict[str, Any]] = None,
    judgment: Optional[str] = None,
    outcome: Optional[str] = None,
    error: Optional[Exception] = None,
) -> None:
    """Print one completed API interaction (safe with tqdm)."""
    parts = ["[verbose]", f"instance={instance_index}"]
    if completion_id:
        parts.append(f"completion_id={completion_id}")
    if error is not None:
        parts.append(f"ERROR={error!r}")
    msg = " ".join(parts)
    tqdm.write(msg)
    if error is None:
        tqdm.write(f"  api_response: {api_response_text!r}")
        if parsed is not None:
            tqdm.write(f"  parsed: {parsed}")
        if judgment is not None and outcome is not None:
            tqdm.write(f"  judgment={judgment} outcome={outcome}")


def _completion_request_body(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    max_completion_tokens: int,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": response_format,
    }
    return body


def _call_gpt_sync(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    debug_state: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str], str]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if debug_http:
                raw = client.with_raw_response.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format,
                )
                if debug_state is not None:
                    hdrs = getattr(raw, "headers", None) or getattr(raw.http_response, "headers", None)
                    _merge_rate_limit_headers(debug_state.setdefault("last_rate_limit_headers", {}), hdrs)
                    rid = None
                    try:
                        rid = raw.http_response.headers.get("x-request-id")
                    except Exception:
                        pass
                    if rid:
                        debug_state.setdefault("sample_request_ids", []).append(rid)
                        debug_state["sample_request_ids"] = debug_state["sample_request_ids"][-8:]
                resp = raw.parse()
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format,
                    reasoning={
                        "effort": "low"
                    }
                )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_content_json(content, legacy_regex)
            req_id = getattr(resp, "id", None)
            return parsed, str(req_id) if req_id else None, content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GPT call failed after {max_retries} attempts: {last_err}")


async def _call_gpt_async(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    debug_lock: asyncio.Lock,
    debug_state: Optional[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> Tuple[Dict[str, Any], Optional[str], str]:
    last_err: Optional[Exception] = None
    async with semaphore:
        for attempt in range(max_retries):
            try:
                if debug_http:
                    raw = await client.with_raw_response.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                        max_completion_tokens=max_completion_tokens,
                        response_format=response_format,
                    )
                    async with debug_lock:
                        if debug_state is not None:
                            hdrs = getattr(raw, "headers", None) or getattr(raw.http_response, "headers", None)
                            _merge_rate_limit_headers(debug_state.setdefault("last_rate_limit_headers", {}), hdrs)
                            try:
                                rid = raw.http_response.headers.get("x-request-id")
                                if rid:
                                    debug_state.setdefault("sample_request_ids", []).append(rid)
                                    debug_state["sample_request_ids"] = debug_state["sample_request_ids"][-8:]
                            except Exception:
                                pass
                    resp = raw.parse()
                else:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                        max_completion_tokens=max_completion_tokens,
                        response_format=response_format,
                    )
                content = (resp.choices[0].message.content or "").strip()
                parsed = _parse_content_json(content, legacy_regex)
                req_id = getattr(resp, "id", None)
                return parsed, str(req_id) if req_id else None, content
            except Exception as e:  # noqa: BLE001
                last_err = e
                await asyncio.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GPT call failed after {max_retries} attempts: {last_err}")


def run_sync(
    input_path: str,
    output_path: str,
    gpt_model: str,
    max_instances: Optional[int],
    start: int,
    end: Optional[int],
    structured: str,
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    verbose: bool,
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if _OPENAI_IMPORT_ERROR is not None or OpenAI is None:
        raise RuntimeError("Install the openai package (pip install openai>=1.0.0)") from _OPENAI_IMPORT_ERROR

    with open(input_path, "r") as f:
        data = json.load(f)

    run_args = data.get("args") or {}
    input_type = str(run_args.get("input_type", "inconsistent"))
    modality_to_report = str(run_args.get("modality_to_report", "image"))

    instances: List[Dict[str, Any]] = data.get("instances") or []
    slice_end = len(instances) if end is None else min(end, len(instances))
    slice_start = max(0, start)
    sliced = instances[slice_start:slice_end]
    if max_instances is not None:
        sliced = sliced[:max_instances]

    response_format = _response_format(structured, input_type)
    client = _make_openai_sync(api_key, base_url=base_url, default_headers=default_headers)
    debug_state: Optional[Dict[str, Any]] = {"last_rate_limit_headers": {}, "sample_request_ids": []} if debug_http else None

    counts = {"correct": 0, "misled": 0, "neither": 0}
    judgment_hist: Dict[str, int] = {"image": 0, "text": 0, "neither": 0}
    n_parse_errors = 0
    n_bert_agree = 0
    n_bert_compare = 0
    instance_records: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for offset, inst in enumerate(tqdm(sliced, desc="GPT judge (sync)")):
        instance_index = slice_start + offset
        tl = inst.get("true_label") or {}
        messages = _build_messages(
            input_type=input_type,
            modality_to_report=modality_to_report,
            prediction=str(inst.get("model_prediction", "")),
            image_captions=list(tl.get("image_captions") or []),
            paired_caption=tl.get("paired_caption"),
        )
        parsed: Optional[Dict[str, Any]] = None
        req_id: Optional[str] = None
        raw_content: Optional[str] = None
        gpt_error: Optional[str] = None
        try:
            parsed, req_id, raw_content = _call_gpt_sync(
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
        except Exception as e:  # noqa: BLE001
            gpt_error = str(e)
            n_parse_errors += 1
            judgment, outcome = "neither", "neither"
            if verbose:
                _verbose_log(instance_index, completion_id=None, api_response_text="", error=e)
        else:
            try:
                judgment, outcome = _gpt_label_to_outcome(input_type, modality_to_report, parsed)
                if verbose:
                    _verbose_log(
                        instance_index,
                        completion_id=req_id,
                        api_response_text=raw_content or "",
                        parsed=parsed,
                        judgment=judgment,
                        outcome=outcome,
                    )
            except Exception as e:  # noqa: BLE001
                gpt_error = str(e)
                n_parse_errors += 1
                judgment, outcome = "neither", "neither"
                if verbose:
                    _verbose_log(
                        instance_index,
                        completion_id=req_id,
                        api_response_text=raw_content or "",
                        parsed=parsed,
                        error=e,
                    )

        judgment_hist[judgment] = judgment_hist.get(judgment, 0) + 1
        _accumulate_counts(counts, outcome)

        bert_label = inst.get("judge_label")
        if isinstance(bert_label, str):
            n_bert_compare += 1
            if bert_label == outcome:
                n_bert_agree += 1

        instance_records.append(
            _instance_eval_record(
                instance_index,
                inst,
                gpt_parsed=parsed,
                gpt_raw_response=raw_content,
                gpt_completion_id=req_id,
                gpt_error=gpt_error,
                judgment=judgment,
                outcome=outcome,
            )
        )

    elapsed = time.perf_counter() - t0

    bert_baseline = {
        "correct": data.get("correct"),
        "misled": data.get("misled"),
        "neither": data.get("neither"),
    }

    out: Dict[str, Any] = {
        "source_result_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "eval_mode": "sync",
        "api_base_url": base_url,
        "gpt_model": gpt_model,
        "structured_output": structured,
        "run_args": run_args,
        "sample_query": data.get("sample_query"),
        "eval_slice": {"start": slice_start, "end": slice_start + len(sliced), "n": len(sliced)},
        "bert_baseline_counts": bert_baseline,
        "gpt_judge_counts": counts,
        "gpt_judgment_hist": judgment_hist,
        "n_parse_errors": n_parse_errors,
        "bert_vs_gpt_agreement": {
            "n_compared": n_bert_compare,
            "n_agree": n_bert_agree,
            "rate": (n_bert_agree / n_bert_compare) if n_bert_compare else None,
        },
        "wall_time_seconds": round(elapsed, 3),
        "instances": instance_records,
    }
    if debug_state:
        out["debug_http"] = {
            "sample_request_ids": debug_state.get("sample_request_ids", []),
            "last_rate_limit_headers": debug_state.get("last_rate_limit_headers", {}),
        }
    return out


async def run_async_meta_one(
    i: int,
    inst: Dict[str, Any],
    *,
    instance_index: int,
    input_type: str,
    modality_to_report: str,
    client: Any,
    gpt_model: str,
    response_format: Dict[str, Any],
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    debug_lock: asyncio.Lock,
    debug_state: Optional[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    verbose: bool,
    verbose_lock: asyncio.Lock,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    tl = inst.get("true_label") or {}
    messages = _build_messages(
        input_type=input_type,
        modality_to_report=modality_to_report,
        prediction=str(inst.get("model_prediction", "")),
        image_captions=list(tl.get("image_captions") or []),
        paired_caption=tl.get("paired_caption"),
    )
    try:
        parsed, req_id, raw_content = await _call_gpt_async(
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
            semaphore,
        )
        if verbose:
            try:
                j, o = _gpt_label_to_outcome(input_type, modality_to_report, parsed)
                async with verbose_lock:
                    _verbose_log(
                        instance_index,
                        completion_id=req_id,
                        api_response_text=raw_content,
                        parsed=parsed,
                        judgment=j,
                        outcome=o,
                    )
            except Exception as e:  # noqa: BLE001
                async with verbose_lock:
                    _verbose_log(
                        instance_index,
                        completion_id=req_id,
                        api_response_text=raw_content,
                        parsed=parsed,
                        error=e,
                    )
        return i, parsed, raw_content, req_id, None
    except Exception as e:  # noqa: BLE001
        if verbose:
            async with verbose_lock:
                _verbose_log(instance_index, completion_id=None, api_response_text="", error=e)
        return i, None, None, None, str(e)


async def run_async(
    input_path: str,
    output_path: str,
    gpt_model: str,
    max_instances: Optional[int],
    start: int,
    end: Optional[int],
    structured: str,
    max_completion_tokens: int,
    max_retries: int,
    legacy_regex: bool,
    debug_http: bool,
    verbose: bool,
    concurrency: int,
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if _OPENAI_IMPORT_ERROR is not None or AsyncOpenAI is None:
        raise RuntimeError("Install the openai package (pip install openai>=1.0.0)") from _OPENAI_IMPORT_ERROR

    with open(input_path, "r") as f:
        data = json.load(f)

    run_args = data.get("args") or {}
    input_type = str(run_args.get("input_type", "inconsistent"))
    modality_to_report = str(run_args.get("modality_to_report", "image"))

    instances: List[Dict[str, Any]] = data.get("instances") or []
    slice_end = len(instances) if end is None else min(end, len(instances))
    slice_start = max(0, start)
    sliced = instances[slice_start:slice_end]
    if max_instances is not None:
        sliced = sliced[:max_instances]

    response_format = _response_format(structured, input_type)
    client = _make_openai_async(api_key, base_url=base_url, default_headers=default_headers)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    debug_lock = asyncio.Lock()
    debug_state: Optional[Dict[str, Any]] = {"last_rate_limit_headers": {}, "sample_request_ids": []} if debug_http else None
    verbose_lock = asyncio.Lock()

    t0 = time.perf_counter()
    tasks = [
        run_async_meta_one(
            i,
            inst,
            instance_index=slice_start + i,
            input_type=input_type,
            modality_to_report=modality_to_report,
            client=client,
            gpt_model=gpt_model,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
            legacy_regex=legacy_regex,
            debug_http=debug_http,
            debug_lock=debug_lock,
            debug_state=debug_state,
            semaphore=semaphore,
            verbose=verbose,
            verbose_lock=verbose_lock,
        )
        for i, inst in enumerate(sliced)
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="GPT judge (async)")
    elapsed = time.perf_counter() - t0

    meta_by_idx: Dict[int, Dict[str, Any]] = {}
    for i, parsed, raw, rid, api_err in results:
        meta_by_idx[i] = {"parsed": parsed, "raw": raw, "rid": rid, "api_error": api_err}

    counts = {"correct": 0, "misled": 0, "neither": 0}
    judgment_hist: Dict[str, int] = {"image": 0, "text": 0, "neither": 0}
    n_parse_errors = 0
    n_bert_agree = 0
    n_bert_compare = 0
    instance_records: List[Dict[str, Any]] = []

    for i, inst in enumerate(sliced):
        instance_index = slice_start + i
        m = meta_by_idx[i]
        parsed = m["parsed"]
        raw_content = m["raw"]
        req_id = m["rid"]
        gpt_error: Optional[str] = m["api_error"]
        try:
            if parsed is None:
                raise ValueError("missing")
            judgment, outcome = _gpt_label_to_outcome(input_type, modality_to_report, parsed)
        except Exception as e:  # noqa: BLE001
            n_parse_errors += 1
            judgment, outcome = "neither", "neither"
            if gpt_error is None:
                gpt_error = str(e)

        judgment_hist[judgment] = judgment_hist.get(judgment, 0) + 1
        _accumulate_counts(counts, outcome)

        bert_label = inst.get("judge_label")
        if isinstance(bert_label, str):
            n_bert_compare += 1
            if bert_label == outcome:
                n_bert_agree += 1

        instance_records.append(
            _instance_eval_record(
                instance_index,
                inst,
                gpt_parsed=parsed,
                gpt_raw_response=raw_content,
                gpt_completion_id=req_id,
                gpt_error=gpt_error,
                judgment=judgment,
                outcome=outcome,
            )
        )

    bert_baseline = {
        "correct": data.get("correct"),
        "misled": data.get("misled"),
        "neither": data.get("neither"),
    }

    out: Dict[str, Any] = {
        "source_result_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "eval_mode": "async",
        "api_base_url": base_url,
        "concurrency": concurrency,
        "gpt_model": gpt_model,
        "structured_output": structured,
        "run_args": run_args,
        "sample_query": data.get("sample_query"),
        "eval_slice": {"start": slice_start, "end": slice_start + len(sliced), "n": len(sliced)},
        "bert_baseline_counts": bert_baseline,
        "gpt_judge_counts": counts,
        "gpt_judgment_hist": judgment_hist,
        "n_parse_errors": n_parse_errors,
        "bert_vs_gpt_agreement": {
            "n_compared": n_bert_compare,
            "n_agree": n_bert_agree,
            "rate": (n_bert_agree / n_bert_compare) if n_bert_compare else None,
        },
        "wall_time_seconds": round(elapsed, 3),
        "instances": instance_records,
    }
    if debug_state:
        out["debug_http"] = {
            "sample_request_ids": debug_state.get("sample_request_ids", []),
            "last_rate_limit_headers": debug_state.get("last_rate_limit_headers", {}),
        }
    return out


def run_batch(
    input_path: str,
    output_path: str,
    gpt_model: str,
    max_instances: Optional[int],
    start: int,
    end: Optional[int],
    structured: str,
    max_completion_tokens: int,
    legacy_regex: bool,
    poll_seconds: float,
    verbose: bool,
    api_key: str,
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if _OPENAI_IMPORT_ERROR is not None or OpenAI is None:
        raise RuntimeError("Install the openai package (pip install openai>=1.0.0)") from _OPENAI_IMPORT_ERROR

    with open(input_path, "r") as f:
        data = json.load(f)

    run_args = data.get("args") or {}
    input_type = str(run_args.get("input_type", "inconsistent"))
    modality_to_report = str(run_args.get("modality_to_report", "image"))

    instances: List[Dict[str, Any]] = data.get("instances") or []
    slice_end = len(instances) if end is None else min(end, len(instances))
    slice_start = max(0, start)
    sliced = instances[slice_start:slice_end]
    if max_instances is not None:
        sliced = sliced[:max_instances]

    response_format = _response_format(structured, input_type)
    batch_input_path = f"{output_path}.batch_input.jsonl"
    batch_meta_path = f"{output_path}.batch_job.json"

    client = _make_openai_sync(api_key, base_url=base_url, default_headers=default_headers)
    t0 = time.perf_counter()

    with open(batch_input_path, "w", encoding="utf-8") as bf:
        for i, inst in enumerate(sliced):
            tl = inst.get("true_label") or {}
            messages = _build_messages(
                input_type=input_type,
                modality_to_report=modality_to_report,
                prediction=str(inst.get("model_prediction", "")),
                image_captions=list(tl.get("image_captions") or []),
                paired_caption=tl.get("paired_caption"),
            )
            body = _completion_request_body(gpt_model, messages, response_format, max_completion_tokens)
            line = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            bf.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(batch_input_path, "rb") as bf:
        batch_file = client.files.create(file=bf, purpose="batch")

    if verbose:
        tqdm.write(f"[verbose] Uploaded batch input file id={batch_file.id} path={batch_input_path!r}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"source": os.path.basename(input_path)},
    )

    meta = {
        "batch_id": batch.id,
        "input_file_id": batch_file.id,
        "batch_input_path": os.path.abspath(batch_input_path),
        "status": batch.status,
    }
    with open(batch_meta_path, "w") as jf:
        json.dump(meta, jf, indent=2)

    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal:
        time.sleep(poll_seconds)
        batch = client.batches.retrieve(batch.id)
        msg = f"Batch {batch.id} status={batch.status!r} counts={getattr(batch, 'request_counts', None)}"
        if verbose:
            tqdm.write(f"[verbose] {msg}")
        else:
            tqdm.write(msg)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status!r} errors={batch.errors}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but output_file_id is missing")

    file_response = client.files.content(batch.output_file_id)
    output_text = file_response.text
    batch_output_path = f"{output_path}.batch_output.jsonl"
    with open(batch_output_path, "w", encoding="utf-8") as out_f:
        out_f.write(output_text)

    idx_meta: Dict[int, Dict[str, Any]] = {}
    for line in output_text.strip().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id")
        if cid is None:
            continue
        idx = int(cid)
        err = row.get("error")
        if err:
            idx_meta[idx] = {
                "parsed": None,
                "raw": None,
                "completion_id": None,
                "error": f"batch_error:{err!r}",
            }
            if verbose:
                tqdm.write(f"[verbose] instance={slice_start + idx} batch_error={err!r}")
            continue
        resp = row.get("response") or {}
        if resp.get("status_code") != 200:
            idx_meta[idx] = {
                "parsed": None,
                "raw": None,
                "completion_id": None,
                "error": f"http_{resp.get('status_code')}",
            }
            if verbose:
                tqdm.write(
                    f"[verbose] instance={slice_start + idx} http_status={resp.get('status_code')!r} body_snip={str(resp)[:200]!r}"
                )
            continue
        body = resp.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            idx_meta[idx] = {
                "parsed": None,
                "raw": None,
                "completion_id": body.get("id"),
                "error": "no_choices_in_batch_response",
            }
            continue
        content = (choices[0].get("message") or {}).get("content") or ""
        completion_id = body.get("id")
        raw_s = str(content).strip()
        if verbose:
            tqdm.write(
                f"[verbose] instance={slice_start + idx} completion_id={completion_id!r} api_response={raw_s!r}"
            )
        try:
            parsed = _parse_content_json(raw_s, legacy_regex)
            idx_meta[idx] = {
                "parsed": parsed,
                "raw": raw_s,
                "completion_id": completion_id,
                "error": None,
            }
            if verbose:
                try:
                    j, o = _gpt_label_to_outcome(input_type, modality_to_report, parsed)
                    tqdm.write(f"  parsed={parsed} judgment={j} outcome={o}")
                except Exception as ex:  # noqa: BLE001
                    tqdm.write(f"  parsed={parsed} (outcome error: {ex!r})")
        except Exception:
            idx_meta[idx] = {
                "parsed": None,
                "raw": raw_s,
                "completion_id": completion_id,
                "error": "json_parse_failed",
            }
            if verbose:
                tqdm.write("  (JSON parse failed)")

    elapsed = time.perf_counter() - t0

    counts = {"correct": 0, "misled": 0, "neither": 0}
    judgment_hist: Dict[str, int] = {"image": 0, "text": 0, "neither": 0}
    n_parse_errors = 0
    n_bert_agree = 0
    n_bert_compare = 0
    instance_records: List[Dict[str, Any]] = []

    for i, inst in enumerate(sliced):
        instance_index = slice_start + i
        m = idx_meta.get(
            i,
            {
                "parsed": None,
                "raw": None,
                "completion_id": None,
                "error": "missing_batch_output_line",
            },
        )
        parsed = m.get("parsed")
        raw_content = m.get("raw")
        req_id = m.get("completion_id")
        gpt_error: Optional[str] = m.get("error")
        try:
            if parsed is None:
                raise ValueError("missing")
            judgment, outcome = _gpt_label_to_outcome(input_type, modality_to_report, parsed)
        except Exception as e:  # noqa: BLE001
            n_parse_errors += 1
            judgment, outcome = "neither", "neither"
            if gpt_error is None:
                gpt_error = str(e)

        judgment_hist[judgment] = judgment_hist.get(judgment, 0) + 1
        _accumulate_counts(counts, outcome)

        bert_label = inst.get("judge_label")
        if isinstance(bert_label, str):
            n_bert_compare += 1
            if bert_label == outcome:
                n_bert_agree += 1

        instance_records.append(
            _instance_eval_record(
                instance_index,
                inst,
                gpt_parsed=parsed,
                gpt_raw_response=raw_content if isinstance(raw_content, str) else None,
                gpt_completion_id=str(req_id) if req_id is not None else None,
                gpt_error=gpt_error,
                judgment=judgment,
                outcome=outcome,
            )
        )

    bert_baseline = {
        "correct": data.get("correct"),
        "misled": data.get("misled"),
        "neither": data.get("neither"),
    }

    out: Dict[str, Any] = {
        "source_result_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "eval_mode": "batch",
        "api_base_url": base_url,
        "batch_id": batch.id,
        "batch_input_path": os.path.abspath(batch_input_path),
        "batch_output_path": os.path.abspath(batch_output_path),
        "batch_job_meta_path": os.path.abspath(batch_meta_path),
        "gpt_model": gpt_model,
        "structured_output": structured,
        "run_args": run_args,
        "sample_query": data.get("sample_query"),
        "eval_slice": {"start": slice_start, "end": slice_start + len(sliced), "n": len(sliced)},
        "bert_baseline_counts": bert_baseline,
        "gpt_judge_counts": counts,
        "gpt_judgment_hist": judgment_hist,
        "n_parse_errors": n_parse_errors,
        "bert_vs_gpt_agreement": {
            "n_compared": n_bert_compare,
            "n_agree": n_bert_agree,
            "rate": (n_bert_agree / n_bert_compare) if n_bert_compare else None,
        },
        "wall_time_seconds": round(elapsed, 3),
        "instances": instance_records,
    }
    return out


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT LLM-as-judge over behavioral evaluation JSON.")
    p.add_argument("--work_dir", type=str, default=os.environ.get("WORKDIR", "."))
    p.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Full path to source results JSON. If set, path-construction flags are ignored. "
        "Default output path mirrors this path with results/ → results_gpt_eval/ (unless --output_json).",
    )
    p.add_argument("--modify_inputs", type=str, default="none")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--dataset", type=str, default="mscoco")
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
        help="Only used to build --input_json path when --input_json is omitted.",
    )
    p.add_argument(
        "--prompt_format",
        type=str,
        default="image_caption",
        choices=["image_caption", "image_document", "image_text"],
    )
    p.add_argument("--order", type=str, default="icq")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gpt_model",
        type=str,
        default=PINNED_DEFAULT_MODEL,
        help=f"Pinned model id recommended (default: {PINNED_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="async",
        choices=["async", "sync", "batch"],
        help="async: concurrent online API; batch: OpenAI Batch API (offline, ~24h window).",
    )
    p.add_argument("--concurrency", type=int, default=32, help="Max in-flight requests (async mode only).")
    p.add_argument(
        "--structured_output",
        type=str,
        default="json_schema",
        choices=["json_schema", "json_object"],
        help="json_schema: strict schema + smaller output; json_object: broader model support.",
    )
    p.add_argument("--max_completion_tokens", type=int, default=128)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument(
        "--legacy_json_parse",
        action="store_true",
        help="If JSON decode fails, strip a {...} substring via regex (old fallback).",
    )
    p.add_argument(
        "--debug_http",
        action="store_true",
        help="Log x-request-id samples and last x-ratelimit-* headers (sync/async).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each API response text (and parsed judgment) as requests complete.",
    )
    p.add_argument("--batch_poll_seconds", type=float, default=15.0)
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--start", type=int, default=0, help="Start index into instances (inclusive).")
    p.add_argument("--end", type=int, default=None, help="End index into instances (exclusive).")
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path. Default: if --input_json is under .../results/..., same path with "
        "results_gpt_eval; else built from --work_dir and path flags.",
    )
    p.add_argument(
        "--env_file",
        type=str,
        default=None,
        help="Path to dotenv file (default: {work_dir}/.env if present, else python-dotenv search from cwd).",
    )
    p.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (e.g. https://openrouter.ai/api/v1). Default: official OpenAI.",
    )
    p.add_argument(
        "--openrouter",
        action="store_true",
        help=f"Shortcut: set base URL to {OPENROUTER_DEFAULT_BASE_URL} (use OPENROUTER_API_KEY).",
    )
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
            "No API key: set OPENAI_API_KEY (OpenAI) or OPENROUTER_API_KEY (OpenRouter), "
            "or add it to a .env file "
            f"(pip install python-dotenv; default: {wd}/.env or use --env_file). "
            "If you use --openrouter, define OPENROUTER_API_KEY (not only OPENAI_API_KEY)."
        )

    default_headers = _optional_openrouter_headers()

    if args.mode == "batch" and base_url and "openrouter.ai" in base_url:
        raise RuntimeError(
            "OpenAI Batch API is not available on OpenRouter; use --mode async or --mode sync."
        )

    if args.input_json:
        input_path = os.path.abspath(args.input_json)
    else:
        input_path = _behavioral_json_path(
            work_dir=os.path.abspath(args.work_dir),
            modify=args.modify_inputs,
            model_name=args.model_name,
            dataset=args.dataset,
            input_type=args.input_type,
            modality=args.modality_to_report,
            prompt_format=args.prompt_format,
            order=args.order,
            seed=args.seed,
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing input JSON: {input_path}")

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
    elif args.input_json:
        derived = _gpt_output_path_from_input_json(input_path)
        if derived is not None:
            output_path = derived
        else:
            output_path = _gpt_output_path(
                work_dir=os.path.abspath(args.work_dir),
                modify=args.modify_inputs,
                model_name=args.model_name,
                dataset=args.dataset,
                input_type=args.input_type,
                modality=args.modality_to_report,
                prompt_format=args.prompt_format,
                order=args.order,
                seed=args.seed,
            )
    else:
        output_path = _gpt_output_path(
            work_dir=os.path.abspath(args.work_dir),
            modify=args.modify_inputs,
            model_name=args.model_name,
            dataset=args.dataset,
            input_type=args.input_type,
            modality=args.modality_to_report,
            prompt_format=args.prompt_format,
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

    if args.mode == "async":
        result = asyncio.run(run_async(**online, concurrency=args.concurrency))
    elif args.mode == "sync":
        result = run_sync(**online)
    else:
        result = run_batch(
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

#!/usr/bin/env python3
"""
Evaluate saved transformation vectors (from train_transformation_vec.py) on val captioning data.

Writes model generations to JSON for external judging (e.g. GPT). No BERT/sentence-transformer
judge is loaded or run in this script.

If you omit --deltas_path, the path is resolved automatically to match train outputs:

  work_dir/results/train_transformation_vec/<model>/<dataset>/<train_orders_tag>/
      <marker|content|baseline_first|baseline_last>_layer<layer_idx>/seed_<train_seed>/deltas.pt

The marker/content checkpoints train d_im + d_cap; the baseline_first/baseline_last checkpoints
train a single vector (mirrored into both keys) applied at one token position.

Use --train_seed (default: same as --seed) and --train_orders (default: icq ciq) to match the
training run.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))

from train_transformation_vec import (
    DEFAULT_TRAIN_ORDERS,
    _inputs_to_device,
    get_spans_for_span_type,
)
from utils.args_utils import get_model_family
from utils.model_utils import load_model_and_preprocess
from utils.patching_utils import (
    get_dataset_with_target_modality_minimal_pairs,
    get_prompt_args,
    prepare_model_inputs,
)
from utils.span_intervention_utils import get_num_hidden_layers
from data.captioning_datasets import _format_captions, _to_caption_strings


def make_hook_multi_from_pairs(pairs):
    """
    pairs: list of (d_im, d_cap, im_sp, cap_sp).
    Sums patches from each pair (e.g. marker deltas at marker spans + content deltas at content spans).
    """

    def hook(module, args):
        h = args[0]
        patch = torch.zeros_like(h, device=h.device, dtype=h.dtype)
        for d_im, d_cap, im_sp, cap_sp in pairs:
            d_im = d_im.to(device=h.device, dtype=h.dtype)
            d_cap = d_cap.to(device=h.device, dtype=h.dtype)
            im_list = list(im_sp) if not isinstance(im_sp, list) else list(im_sp)
            cap_list = list(cap_sp) if not isinstance(cap_sp, list) else list(cap_sp)
            for i in im_list:
                if i < h.shape[1]:
                    patch[:, i, :] += d_im
            for i in cap_list:
                if i < h.shape[1]:
                    patch[:, i, :] += d_cap
        return (h + patch,) + args[1:]

    return hook


def register_intervention_hooks(_layers_eval, delta_specs):
    """
    delta_specs: list of dicts with keys delta_im_by_layer, delta_cap_by_layer, im_sp, cap_sp.
    For each layer index present in any spec, registers one hook that sums all applicable patches.
    """
    layer_keys = set()
    for spec in delta_specs:
        layer_keys.update(spec["delta_im_by_layer"].keys())
    handles = []
    for l in sorted(layer_keys):
        pairs = []
        for spec in delta_specs:
            if l in spec["delta_im_by_layer"]:
                pairs.append(
                    (
                        spec["delta_im_by_layer"][l],
                        spec["delta_cap_by_layer"][l],
                        spec["im_sp"],
                        spec["cap_sp"],
                    )
                )
        if pairs:
            h = _layers_eval[l].register_forward_pre_hook(make_hook_multi_from_pairs(pairs))
            handles.append(h)
    return handles


def generate_with_intervention(
    raw_model,
    tokenizer,
    inputs,
    _layers_eval,
    delta_specs,
):
    """
    delta_specs: list of one dict (marker or content) or two dicts (marker + content checkpoints).
    """
    handles = register_intervention_hooks(_layers_eval, delta_specs)
    try:
        with torch.no_grad():
            outputs = raw_model.generate(
                **{k: v for k, v in inputs.items() if k != "labels"},
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        for h in handles:
            h.remove()
    pred = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return pred


def generate_baseline(raw_model, tokenizer, inputs):
    with torch.no_grad():
        outputs = raw_model.generate(
            **{k: v for k, v in inputs.items() if k != "labels"},
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    pred = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return pred


def load_delta_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if "delta_im_by_layer" not in ckpt or "delta_cap_by_layer" not in ckpt:
        raise ValueError(f"Expected delta_im_by_layer / delta_cap_by_layer in {path}")
    return ckpt["delta_im_by_layer"], ckpt["delta_cap_by_layer"]


def default_train_deltas_pt(work_dir, model_name, dataset, train_orders, span_type, layer_idx, train_seed):
    """Path to deltas.pt written by train_transformation_vec.py."""
    orders_tag = "_".join(train_orders)
    span_type_layer = f"{span_type}_layer{layer_idx}"
    return os.path.join(
        work_dir,
        "results",
        "train_transformation_vec",
        model_name.replace("/", "_"),
        dataset,
        orders_tag,
        span_type_layer,
        f"seed_{train_seed}",
        "deltas.pt",
    )


def resolve_delta_paths(args, layer_idx):
    """
    Use explicit --deltas_path if set; otherwise default_train_deltas_pt for the chosen
    intervention (which is also the span_type used in the train output layout).
    train_seed: --train_seed if set, else --seed (matches training checkpoint folder).
    train_orders: --train_orders if set, else DEFAULT_TRAIN_ORDERS.
    """
    train_orders = (
        list(args.train_orders)
        if args.train_orders is not None
        else list(DEFAULT_TRAIN_ORDERS)
    )
    train_seed = args.train_seed if args.train_seed is not None else args.seed

    path = args.deltas_path or default_train_deltas_pt(
        args.work_dir, args.model_name, args.dataset, train_orders, args.intervention, layer_idx, train_seed
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing deltas checkpoint:\n  {path}\n"
            f"Expected train_transformation_vec layout with orders_tag={'_'.join(train_orders)!r}, "
            f"span_type={args.intervention!r}, layer_idx={layer_idx}, train_seed={train_seed}. "
            f"Pass --deltas_path or adjust --train_orders / --train_seed / --layer_depth."
        )
    return {"single": path, "train_orders": train_orders, "train_seed": train_seed}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate transformation vectors (captioning)")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["mscoco", "flickr30k"])
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--work_dir", type=str, default=None)
    p.add_argument(
        "--intervention",
        type=str,
        required=True,
        choices=["marker", "content", "baseline_first", "baseline_last"],
        help="Which trained deltas to apply: marker/content (d_im + d_cap at marker/content spans) "
        "or baseline_first/baseline_last (single vector at one token position).",
    )
    p.add_argument(
        "--deltas_path",
        type=str,
        default=None,
        help="Path to deltas.pt. If omitted, uses train_transformation_vec default layout for the chosen intervention.",
    )
    p.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help="Training run seed (seed_* folder under train_transformation_vec). Default: same as --seed.",
    )
    p.add_argument(
        "--train_orders",
        type=str,
        nargs="+",
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
        help="Training train_orders tag for auto paths (joined with _). Default: icq ciq (same as train default).",
    )
    p.add_argument(
        "--order",
        type=str,
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
    )
    p.add_argument(
        "--eval_orders",
        type=str,
        nargs="+",
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
    )
    p.add_argument("--num_eval_samples", type=int, default=100)
    p.add_argument(
        "--layer_depth",
        type=float,
        default=0.0,
        help="Same as training: maps to layer index via round(layer_depth * (N_LAYERS-1)).",
    )
    p.add_argument("--prompt_format", type=str, default="image_caption", choices=["image_caption", "image_document"])
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Eval split (default val, same as training eval).",
    )
    p.add_argument(
        "--mscoco_split",
        type=str,
        default="val",
        choices=["val", "train"],
    )
    p.add_argument(
        "--num_save_examples",
        type=int,
        default=None,
        help="Max response rows in JSON (default: all collected samples = num_eval_pairs).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: work_dir/results/eval_transformation_vec/...).",
    )
    return p.parse_args()


def resolve_eval_orders(args):
    """Resolve eval order list (same defaults as train_transformation_vec)."""
    if args.eval_orders is not None:
        return list(args.eval_orders)
    if args.order is not None:
        return [args.order]
    return list(DEFAULT_TRAIN_ORDERS)


def main():
    args = parse_args()
    args.work_dir = args.work_dir or _project_root

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_orders = resolve_eval_orders(args)

    pmpt_args = get_prompt_args(
        args.model_name,
        args.dataset,
        order=eval_orders[0],
        work_dir=args.work_dir,
        mscoco_split=args.mscoco_split,
        prompt_format=args.prompt_format,
    )
    model_family = get_model_family(pmpt_args)
    pmpt_args.model_family = model_family

    print(f"Loading model {args.model_name}...")
    model, processor = load_model_and_preprocess(pmpt_args)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    print(f"Device: {device}")

    for p in model.parameters():
        p.requires_grad = False

    raw_model = model
    try:
        _layers_eval = raw_model.language_model.layers
    except AttributeError:
        _layers_eval = raw_model.layers

    N_LAYERS = get_num_hidden_layers(model)
    if not (0 <= args.layer_depth <= 1):
        raise ValueError(f"layer_depth must be in [0, 1], got {args.layer_depth}")
    layer_idx = int(round(args.layer_depth * (N_LAYERS - 1))) if N_LAYERS > 1 else 0
    layer_idx = max(0, min(layer_idx, N_LAYERS - 1))
    print(f"Using layer_idx={layer_idx} (layer_depth={args.layer_depth}, N_LAYERS={N_LAYERS})")

    resolved = resolve_delta_paths(args, layer_idx)
    args.deltas_path = resolved["single"]
    print(
        f"Resolved checkpoint: train_orders={resolved['train_orders']!r}, "
        f"train_seed={resolved['train_seed']}"
    )
    print(f"  {args.deltas_path}")

    is_baseline = args.intervention in ("baseline_first", "baseline_last")
    text_prefix_type = "caption" if args.prompt_format == "image_caption" else "document"

    def _check_layer_keys(dim_by_layer, name):
        if layer_idx not in dim_by_layer:
            raise ValueError(
                f"{name}: layer_idx {layer_idx} not in checkpoint keys {sorted(dim_by_layer.keys())}. "
                "Use the same --layer_depth as training."
            )

    # Load checkpoint onto CPU first; hook moves slices to GPU. Name the loaded vectors
    # by intervention type so the baseline single-vector case doesn't masquerade as d_im/d_cap.
    ckpt_a, ckpt_b = load_delta_checkpoint(args.deltas_path)
    if is_baseline:
        # Baseline trains a single vector applied at one token position. train_transformation_vec
        # mirrors it into both checkpoint keys, so we keep just one and call it delta_baseline.
        delta_baseline_by_layer = ckpt_a
        delta_im_by_layer = delta_cap_by_layer = None
        _check_layer_keys(delta_baseline_by_layer, "baseline deltas checkpoint")
    else:
        delta_baseline_by_layer = None
        delta_im_by_layer, delta_cap_by_layer = ckpt_a, ckpt_b
        _check_layer_keys(delta_im_by_layer, "deltas checkpoint")

    def build_delta_specs_for_sample(inputs_gpu, intervention_mode):
        """Returns list of spec dicts for register_intervention_hooks.

        For marker/content the span is two groups (im_sp + cap_sp) with d_im / d_cap.
        For baseline_first/baseline_last the span is a SINGLE token position: im_sp holds
        that one index and cap_sp is empty, so only the single delta_baseline vector is
        applied at that one position — matching the single-vector training.
        """
        im_sp, cap_sp = get_spans_for_span_type(
            inputs_gpu, intervention_mode, model_family, tokenizer, text_prefix_type
        )
        if is_baseline:
            if not (len(im_sp) == 1 and len(cap_sp) == 0):
                raise ValueError(
                    f"Expected a single-index baseline span for {intervention_mode}, "
                    f"got im_sp={im_sp}, cap_sp={cap_sp}"
                )
            # cap_sp is empty, so the cap slot is never read; reuse delta_baseline for both.
            return [
                {
                    "delta_im_by_layer": delta_baseline_by_layer,
                    "delta_cap_by_layer": delta_baseline_by_layer,
                    "im_sp": im_sp,
                    "cap_sp": cap_sp,
                }
            ]
        return [
            {
                "delta_im_by_layer": delta_im_by_layer,
                "delta_cap_by_layer": delta_cap_by_layer,
                "im_sp": im_sp,
                "cap_sp": cap_sp,
            }
        ]

    per_order_eval = max(1, args.num_eval_samples // len(eval_orders))
    used_eval_idx = set()
    eval_pairs = []
    for order in eval_orders:
        pmpt_args_e = get_prompt_args(
            args.model_name,
            args.dataset,
            order=order,
            work_dir=args.work_dir,
            mscoco_split=args.mscoco_split,
            prompt_format=args.prompt_format,
        )
        pmpt_args_e.model_family = model_family
        eval_image_dataset = get_dataset_with_target_modality_minimal_pairs(
            pmpt_args_e, "image", split=args.split
        )
        eval_text_dataset = get_dataset_with_target_modality_minimal_pairs(
            pmpt_args_e, "text", split=args.split
        )
        pool_n = len(eval_image_dataset)
        n_target = min(per_order_eval, pool_n)
        available = [i for i in range(pool_n) if i not in used_eval_idx]
        if len(available) < n_target:
            available = list(range(pool_n))
        n_take = min(n_target, len(available))
        eval_indices = random.sample(available, n_take)
        used_eval_idx.update(eval_indices)
        print(f"  eval order={order}: pool={pool_n}, sampling {n_take} indices")
        for idx in tqdm(eval_indices, desc=f"Collecting eval pairs ({order})"):
            sample_im = eval_image_dataset[idx]
            sample_txt = eval_text_dataset[idx]
            inputs_clean = prepare_model_inputs(sample_im, processor, device="cpu")
            inputs_corr = prepare_model_inputs(sample_txt, processor, device="cpu")
            img_caps = _format_captions(_to_caption_strings(sample_im["image_captions"]))
            img_continuation = img_caps[0] if img_caps else ""
            cap_continuation = sample_im["paired_caption"] or ""
            if not img_continuation or not cap_continuation:
                continue
            eval_pairs.append(
                (
                    order,
                    idx,
                    inputs_clean,
                    inputs_corr,
                    sample_im,
                    img_continuation,
                    cap_continuation,
                )
            )

    print(f"Eval pairs: {len(eval_pairs)}")

    responses = []
    max_rows = args.num_save_examples if args.num_save_examples is not None else len(eval_pairs)
    n_to_run = min(len(eval_pairs), max_rows)

    for i in tqdm(range(n_to_run), desc="Generating (baseline + intervention)"):
        order_ev, data_idx, inputs_clean, inputs_corr, sample_im, img_cont, cap_cont = eval_pairs[i]
        img_caps = sample_im["image_captions"]
        paired_cap = sample_im["paired_caption"]
        img_refs = _format_captions(_to_caption_strings(img_caps))

        inputs_clean_gpu = _inputs_to_device(inputs_clean, device)
        inputs_corr_gpu = _inputs_to_device(inputs_corr, device)

        pred_clean = generate_baseline(raw_model, tokenizer, inputs_clean_gpu)
        pred_corr = generate_baseline(raw_model, tokenizer, inputs_corr_gpu)

        specs_clean = build_delta_specs_for_sample(inputs_clean_gpu, args.intervention)
        specs_corr = build_delta_specs_for_sample(inputs_corr_gpu, args.intervention)
        pred_iv_clean = generate_with_intervention(
            raw_model, tokenizer, inputs_clean_gpu, _layers_eval, specs_clean
        )
        pred_iv_corr = generate_with_intervention(
            raw_model, tokenizer, inputs_corr_gpu, _layers_eval, specs_corr
        )

        responses.append(
            {
                "order": order_ev,
                "dataset_index": int(data_idx),
                "prompt_format": args.prompt_format,
                "modality_target_clean_inputs": "image",
                "modality_target_corr_inputs": "text",
                "image_caption_reference": img_refs,
                "paired_caption": paired_cap,
                "pred_baseline_image_target_prompt": pred_clean,
                "pred_baseline_text_target_prompt": pred_corr,
                "pred_intervention_image_target_prompt": pred_iv_clean,
                "pred_intervention_text_target_prompt": pred_iv_corr,
                "image_caption_first_text": img_cont,
                "paired_caption_text": cap_cont,
            }
        )

    print(
        f"\nCollected {len(responses)} response records (requested up to {max_rows} of {len(eval_pairs)} pairs); "
        "no in-script judge — use external eval (e.g. GPT)."
    )

    eval_results = {
        "intervention": args.intervention,
        "train_orders_for_checkpoint": resolved["train_orders"],
        "train_seed_for_checkpoint": resolved["train_seed"],
        "deltas_path": args.deltas_path,
        "eval_orders": eval_orders,
        "split": args.split,
        "mscoco_split": args.mscoco_split,
        "layer_idx": layer_idx,
        "layer_depth": args.layer_depth,
        "num_eval_pairs_built": len(eval_pairs),
        "num_responses_saved": len(responses),
        "num_save_examples_cap": max_rows,
        "note": "predictions only; im2cap/cap2im flip rates require external judge (e.g. GPT).",
        "responses": responses,
        "saved_examples": responses,
    }

    if args.output_dir:
        out_dir = args.output_dir
    else:
        iv_tag = args.intervention
        out_dir = os.path.join(
            args.work_dir,
            "results",
            "eval_transformation_vec",
            args.model_name.replace("/", "_"),
            args.dataset,
            iv_tag,
            f"layer_depth_{args.layer_depth:g}".replace(".", "p"),
            f"split_{args.split}",
            f"seed_{args.train_seed}",
        )
    os.makedirs(out_dir, exist_ok=True)
    eval_path = os.path.join(out_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval results and sample outputs to {eval_path}")


if __name__ == "__main__":
    main()

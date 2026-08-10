#!/usr/bin/env python3
"""
Train transformation vectors (d_im, d_cap per layer) for captioning datasets.
Uses generation loss: reduce loss of clean image continuation when asked for text (with vecs),
and vice versa. Adapted for MSCOCO/Flickr30k inconsistent pairs.

By default trains on both modality orders icq and ciq with num_delta_samples split evenly
(e.g. 4K total -> 2K per order, disjoint indices when the pool is large enough). Eval mirrors
the same orders unless --eval_orders is set.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))

from utils.args_utils import get_model_family
from utils.model_utils import load_model_and_preprocess, load_bert_judge
from utils.parse_spans_utils import get_span
from utils.patching_utils import (
    get_dataset_with_target_modality_minimal_pairs,
    get_prompt_args,
    prepare_model_inputs,
    prepare_model_inputs_with_continuation,
)
from utils.span_intervention_utils import get_num_hidden_layers
from utils.prompt_evaluation_utils import bert_judge_3way
from data.captioning_datasets import _format_captions, _to_caption_strings

# Default modality-token orders when neither --order nor --train_orders is given.
DEFAULT_TRAIN_ORDERS = ("icq", "ciq")


def _inputs_to_device(inputs, device):
    """Move input dict tensors to device (inputs stay on CPU until used)."""
    return {k: v.to(device) if hasattr(v, "to") and v is not None else v for k, v in inputs.items()}


def _baseline_positions(span_type, seq_len):
    """Single token position for the baseline span types.

    baseline_first: the first token (index 0, typically BOS / start of prompt).
    baseline_last: the last token of the given sequence (seq_len - 1). When `seq_len`
    is a prompt length, this is the final prompt token right before generation.
    """
    if span_type == "baseline_first":
        return [0]
    if span_type == "baseline_last":
        return [max(0, seq_len - 1)]
    raise ValueError(f"_baseline_positions called with non-baseline span_type={span_type}")


def get_spans_for_span_type(inputs, span_type, model_family, tokenizer, text_prefix_type):
    """
    Get (im_span_indices, cap_span_indices) for span_type 'marker', 'content',
    'baseline_first' or 'baseline_last'.

    marker: image_start + image_end (2 deltas) + cap_start (1 delta) = 3 deltas total
    content: image_content + cap_content (unchanged)
    baseline_first/baseline_last: a single token position (returned as the image span;
        caption span is empty), so a single trained vector is applied at one position.
        For baseline_last the index is derived from the passed `inputs` length.
    """
    if span_type in ("baseline_first", "baseline_last"):
        seq_len = inputs["input_ids"].shape[1]
        return _baseline_positions(span_type, seq_len), []
    if span_type == "marker":
        im_start = get_span(inputs, "image_start", model_family=model_family, tokenizer=tokenizer, text_prefix_type=text_prefix_type)
        im_end = get_span(inputs, "image_end", model_family=model_family, tokenizer=tokenizer, text_prefix_type=text_prefix_type)
        cap_start = get_span(inputs, "cap_start", model_family=model_family, tokenizer=tokenizer, text_prefix_type=text_prefix_type)
        im_span = list(im_start) + list(im_end)
        cap_span = list(cap_start)
    elif span_type == "content":
        im_span = get_span(inputs, "image_content", model_family=model_family, tokenizer=tokenizer, text_prefix_type=text_prefix_type)
        cap_span = get_span(inputs, "cap_content", model_family=model_family, tokenizer=tokenizer, text_prefix_type=text_prefix_type)
        im_span = list(im_span) if not isinstance(im_span, list) else im_span
        cap_span = list(cap_span) if not isinstance(cap_span, list) else cap_span
    else:
        raise ValueError(
            f"span_type must be 'marker', 'content', 'baseline_first' or 'baseline_last', got {span_type}"
        )
    return im_span, cap_span


def make_hook(d_im_param, d_cap_param, im_sp, cap_sp):
    def hook(module, args):
        h = args[0]
        d_im = d_im_param.to(device=h.device, dtype=h.dtype)
        d_cap = d_cap_param.to(device=h.device, dtype=h.dtype)
        patch = torch.zeros_like(h, device=h.device, dtype=h.dtype)
        im_list = list(im_sp) if not isinstance(im_sp, list) else list(im_sp)
        cap_list = list(cap_sp) if not isinstance(cap_sp, list) else list(cap_sp)
        for i in im_list:
            if i < h.shape[1]:
                patch[:, i, :] = d_im
        for i in cap_list:
            if i < h.shape[1]:
                patch[:, i, :] = d_cap
        return (h + patch,) + args[1:]
    return hook


def make_single_hook(d_param, positions):
    """Hook that adds a single trained vector at one (or few) token position(s).

    Used by the baseline_first / baseline_last span types, which optimize a single
    vector applied at one position in both modality-order directions.
    """
    def hook(module, args):
        h = args[0]
        d = d_param.to(device=h.device, dtype=h.dtype)
        patch = torch.zeros_like(h, device=h.device, dtype=h.dtype)
        pos_list = list(positions) if not isinstance(positions, list) else positions
        for i in pos_list:
            if 0 <= i < h.shape[1]:
                patch[:, i, :] = d
        return (h + patch,) + args[1:]
    return hook


def _resolve_train_eval_orders(args):
    """Resolve modality token orders for train/eval.

    Default (no --order / --train_orders): icq and ciq with num_delta_samples split evenly (e.g. 2K+2K for 4K).
    --order X alone: single-order run (same as before).
    --train_orders a b ...: explicit list (mutually exclusive with --order).
    """
    if args.train_orders is not None and args.order is not None:
        raise ValueError("Use either --train_orders ... or --order ..., not both.")
    if args.train_orders is not None:
        train_orders = list(args.train_orders)
    elif args.order is not None:
        train_orders = [args.order]
    else:
        train_orders = list(DEFAULT_TRAIN_ORDERS)
    eval_orders = list(args.eval_orders) if args.eval_orders is not None else list(train_orders)
    return train_orders, eval_orders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformation vectors with generation loss (captioning datasets)"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["mscoco", "flickr30k"])
    parser.add_argument(
        "--span_type",
        type=str,
        required=True,
        choices=["marker", "content", "baseline_first", "baseline_last"],
        help="marker/content: train d_im + d_cap. baseline_first/baseline_last: train a single "
        "vector applied at one token position (first or last prompt token).",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
        help="Single modality order (legacy). Default dual-order icq+ciq if omitted.",
    )
    parser.add_argument(
        "--train_orders",
        type=str,
        nargs="+",
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
        help=f"Modality orders for training; num_delta_samples is split evenly across orders. Default: {' '.join(DEFAULT_TRAIN_ORDERS)}.",
    )
    parser.add_argument(
        "--eval_orders",
        type=str,
        nargs="+",
        default=None,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
        help="Orders for eval (default: same as training orders). num_eval_samples split evenly.",
    )
    parser.add_argument(
        "--num_delta_samples",
        type=int,
        default=4000,
        help="Total training pairs; split evenly across train_orders (e.g. 4K -> 2K per order for icq+ciq).",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=200,
        help="Total eval pairs; split evenly across eval_orders.",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument(
        "--layer_depth",
        type=float,
        default=0.0,
        help="Layer depth in [0, 1] mapping to a single layer: 0=first, 1=last. Only this layer is trained.",
    )
    parser.add_argument("--prompt_format", type=str, default="image_caption", choices=["image_caption", "image_document"])
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (effective batch = 1 * this)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.work_dir = args.work_dir or _project_root
    is_baseline = args.span_type in ("baseline_first", "baseline_last")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_orders, eval_orders = _resolve_train_eval_orders(args)
    orders_tag = "_".join(train_orders)

    pmpt_args = get_prompt_args(
        args.model_name,
        args.dataset,
        order=train_orders[0],
        work_dir=args.work_dir,
        mscoco_split="train",
        prompt_format=args.prompt_format,
    )
    model_family = get_model_family(pmpt_args)
    pmpt_args.model_family = model_family

    print(f"Loading model {args.model_name}...")
    model, processor = load_model_and_preprocess(pmpt_args)
    tokenizer = processor.tokenizer

    # Use model's device for inputs (critical when device_map="auto" spreads layers across GPUs)
    device = next(model.parameters()).device
    print(f"Device for inputs: {device}")

    # Freeze model params to save memory (we only train delta vectors)
    for p in model.parameters():
        p.requires_grad = False

    N_LAYERS = get_num_hidden_layers(model)
    text_prefix_type = "caption"

    if not (0 <= args.layer_depth <= 1):
        raise ValueError(f"layer_depth must be in [0, 1], got {args.layer_depth}")
    layer_idx = int(round(args.layer_depth * (N_LAYERS - 1))) if N_LAYERS > 1 else 0
    layer_idx = max(0, min(layer_idx, N_LAYERS - 1))
    print(f"Intervening on layer {layer_idx} (layer_depth={args.layer_depth} -> layer {layer_idx}/{N_LAYERS-1})")

    print(f"Model family: {model_family}, N_LAYERS: {N_LAYERS}")
    print(f"Train orders: {train_orders} (num_delta_samples={args.num_delta_samples} total, split evenly)")
    print(f"Eval orders: {eval_orders} (num_eval_samples={args.num_eval_samples} total, split evenly)")
    if args.grad_accum_steps > 1:
        print(f"Gradient accumulation: {args.grad_accum_steps} steps (effective batch size {args.grad_accum_steps})")

    per_order_train = max(1, args.num_delta_samples // len(train_orders))
    used_train_idx = set()
    pairs = []
    for order in train_orders:
        pmpt_args_o = get_prompt_args(
            args.model_name,
            args.dataset,
            order=order,
            work_dir=args.work_dir,
            mscoco_split="train",
            prompt_format=args.prompt_format,
        )
        pmpt_args_o.model_family = model_family
        train_image_dataset = get_dataset_with_target_modality_minimal_pairs(pmpt_args_o, "image", split="train")
        train_text_dataset = get_dataset_with_target_modality_minimal_pairs(pmpt_args_o, "text", split="train")
        pool_n = len(train_image_dataset)
        n_target = min(per_order_train, pool_n)
        available = [i for i in range(pool_n) if i not in used_train_idx]
        if len(available) < n_target:
            available = list(range(pool_n))
        n_take = min(n_target, len(available))
        train_indices = random.sample(available, n_take)
        used_train_idx.update(train_indices)
        print(f"  order={order}: pool={pool_n}, sampling {n_take} indices (disjoint across orders when possible)")
        for idx in tqdm(train_indices, desc=f"Collecting train pairs ({order})"):
            sample_im = train_image_dataset[idx]
            sample_txt = train_text_dataset[idx]
            inputs_clean = prepare_model_inputs(sample_im, processor, device="cpu")
            im_span_idcs, cap_span_idcs = get_spans_for_span_type(
                inputs_clean, args.span_type, model_family, tokenizer, text_prefix_type
            )
            img_caps = _format_captions(_to_caption_strings(sample_im["image_captions"]))
            img_continuation = img_caps[0] if img_caps else ""
            cap_continuation = sample_im["paired_caption"] or ""
            if not img_continuation or not cap_continuation:
                continue
            pairs.append((idx, im_span_idcs, cap_span_idcs, sample_im, sample_txt, img_continuation, cap_continuation))
    random.shuffle(pairs)

    per_order_eval = max(1, args.num_eval_samples // len(eval_orders))
    used_eval_idx = set()
    eval_pairs = []
    for order in eval_orders:
        pmpt_args_e = get_prompt_args(
            args.model_name,
            args.dataset,
            order=order,
            work_dir=args.work_dir,
            mscoco_split="val",
            prompt_format=args.prompt_format,
        )
        pmpt_args_e.model_family = model_family
        eval_image_dataset = get_dataset_with_target_modality_minimal_pairs(pmpt_args_e, "image", split="val")
        eval_text_dataset = get_dataset_with_target_modality_minimal_pairs(pmpt_args_e, "text", split="val")
        pool_n = len(eval_image_dataset)
        n_target = min(per_order_eval, pool_n)
        available = [i for i in range(pool_n) if i not in used_eval_idx]
        if len(available) < n_target:
            available = list(range(pool_n))
        n_take = min(n_target, len(available))
        eval_indices = random.sample(available, n_take)
        used_eval_idx.update(eval_indices)
        print(f"  eval order={order}: pool={pool_n}, sampling {n_take} indices (disjoint across orders when possible)")
        for idx in tqdm(eval_indices, desc=f"Collecting eval pairs ({order})"):
            sample_im = eval_image_dataset[idx]
            sample_txt = eval_text_dataset[idx]
            inputs_clean = prepare_model_inputs(sample_im, processor, device="cpu")
            inputs_corr = prepare_model_inputs(sample_txt, processor, device="cpu")
            im_span_idcs, cap_span_idcs = get_spans_for_span_type(
                inputs_clean, args.span_type, model_family, tokenizer, text_prefix_type
            )
            img_caps = _format_captions(_to_caption_strings(sample_im["image_captions"]))
            img_continuation = img_caps[0] if img_caps else ""
            cap_continuation = sample_im["paired_caption"] or ""
            if not img_continuation or not cap_continuation:
                continue
            eval_pairs.append(
                (order, idx, inputs_clean, inputs_corr, im_span_idcs, cap_span_idcs, sample_im, img_continuation, cap_continuation)
            )

    print(f"Train pairs: {len(pairs)}, Eval pairs: {len(eval_pairs)}")

    raw_model = model
    try:
        _layers = raw_model.language_model.layers
    except AttributeError:
        _layers = raw_model.layers

    HIDDEN_DIM = (
        raw_model.config.hidden_size
        if hasattr(raw_model.config, "hidden_size")
        else raw_model.config.text_config.hidden_size
    )

    layers_to_train = [layer_idx]
    if is_baseline:
        # Single trained vector applied at one token position, shared across both directions.
        delta_params = nn.ParameterList([nn.Parameter(torch.zeros(HIDDEN_DIM, device=device, dtype=torch.float32))])
        delta_im_params = delta_cap_params = None
        opt = torch.optim.Adam(delta_params.parameters(), lr=args.lr)
    else:
        delta_params = None
        delta_im_params = nn.ParameterList([nn.Parameter(torch.zeros(HIDDEN_DIM, device=device, dtype=torch.float32))])
        delta_cap_params = nn.ParameterList([nn.Parameter(torch.zeros(HIDDEN_DIM, device=device, dtype=torch.float32))])
        opt = torch.optim.Adam(list(delta_im_params.parameters()) + list(delta_cap_params.parameters()), lr=args.lr)
    delta_im_by_layer = {}
    delta_cap_by_layer = {}

    handles = [None] * N_LAYERS
    loss_history = []

    # Train with generation loss
    # Loss 1: inputs_corr (ask caption) + intervention -> predict image continuation (reduce loss)
    # Loss 2: inputs_clean (ask image) + intervention -> predict caption continuation (reduce loss)
    grad_accum_steps = args.grad_accum_steps
    opt.zero_grad()
    for epoch in range(args.n_epochs):
        total_loss = 0.0
        n_batches = 0
        recent_losses = []
        for data_idx, im_sp, cap_sp, sample_im, sample_txt, img_cont, cap_cont in tqdm(pairs, desc=f"Epoch {epoch + 1}"):
            im_sp_list = list(im_sp) if not isinstance(im_sp, list) else list(im_sp)
            cap_sp_list = list(cap_sp) if not isinstance(cap_sp, list) else list(cap_sp)

            for h in handles:
                if h is not None:
                    h.remove()

            # Build inputs with continuation for loss
            inputs_corr_with_cont, plen_corr = prepare_model_inputs_with_continuation(sample_txt, processor, img_cont, device=device)
            inputs_clean_with_cont, _ = prepare_model_inputs_with_continuation(sample_im, processor, cap_cont, device=device)

            # The image-ask and caption-ask prompts share the same pre-continuation length, so
            # for baseline the single intervention position is computed once and reused for both passes.
            if is_baseline:
                baseline_pos = _baseline_positions(args.span_type, plen_corr)
            for l in layers_to_train:
                if is_baseline:
                    hook = make_single_hook(delta_params[0], baseline_pos)
                else:
                    hook = make_hook(delta_im_params[0], delta_cap_params[0], im_sp_list, cap_sp_list)
                handles[l] = _layers[l].register_forward_pre_hook(hook)

            # Loss 1: ask caption + vecs -> predict image continuation
            out_corr = raw_model(**inputs_corr_with_cont)
            loss_corr = out_corr.loss if hasattr(out_corr, "loss") else torch.tensor(0.0, device=device)

            # Loss 2: ask image + vecs -> predict caption continuation
            out_clean = raw_model(**inputs_clean_with_cont)
            loss_clean = out_clean.loss if hasattr(out_clean, "loss") else torch.tensor(0.0, device=device)

            loss = (loss_corr + loss_clean) / grad_accum_steps
            loss.backward()

            batch_loss = (loss_corr + loss_clean).item()
            total_loss += batch_loss
            n_batches += 1
            loss_history.append(batch_loss)
            recent_losses.append(batch_loss)

            if n_batches % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()

            if n_batches % 20 == 0:
                cum_avg = total_loss / n_batches
                last20_avg = sum(recent_losses) / len(recent_losses)
                recent_losses = recent_losses[-20:] if len(recent_losses) > 20 else recent_losses
                print(f"  epoch {epoch+1} step {n_batches} cumulative_loss_avg={cum_avg:.4f} last20_loss={last20_avg:.4f}")

        # Step on any remaining accumulated gradients
        if n_batches % grad_accum_steps != 0:
            opt.step()
            opt.zero_grad()
        print(f"Epoch {epoch+1} avg_loss={total_loss/n_batches:.4f}")

    for h in handles:
        if h is not None:
            h.remove()

    for l in layers_to_train:
        if is_baseline:
            # One trained vector; mirror it into both keys so the deltas.pt format is unchanged.
            delta_im_by_layer[l] = delta_params[0].detach().clone()
            delta_cap_by_layer[l] = delta_params[0].detach().clone()
        else:
            delta_im_by_layer[l] = delta_im_params[0].detach().clone()
            delta_cap_by_layer[l] = delta_cap_params[0].detach().clone()

    # Save
    layer_suffix = str(layer_idx)
    span_type_layer = f"{args.span_type}_layer{layer_suffix}"
    output_dir = os.path.join(
        args.work_dir,
        "results",
        "train_transformation_vec",
        args.model_name.replace("/", "_"),
        args.dataset,
        orders_tag,
        span_type_layer,
        "seed_" + str(args.seed),
    )
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "deltas.pt")
    torch.save({"delta_im_by_layer": delta_im_by_layer, "delta_cap_by_layer": delta_cap_by_layer}, out_path)
    print(f"Saved deltas to {out_path}")

    loss_path = os.path.join(output_dir, "loss.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Saved loss history ({len(loss_history)} steps) to {loss_path}")

    # Eval: generation + BERT judge
    try:
        _layers_eval = raw_model.language_model.layers
    except AttributeError:
        _layers_eval = raw_model.layers

    encoder = load_bert_judge(getattr(pmpt_args, "judge_model", "sentence-transformers/all-mpnet-base-v2"))
    judge_margin = getattr(pmpt_args, "judge_margin", 0.1)
    judge_threshold = getattr(pmpt_args, "judge_threshold", 0.4)

    def generate_and_judge(inputs, im_sp, cap_sp, delta_im_by_layer, delta_cap_by_layer):
        eval_handles = []
        for l in delta_im_by_layer.keys():
            h = _layers_eval[l].register_forward_pre_hook(
                make_hook(delta_im_by_layer[l], delta_cap_by_layer[l], im_sp, cap_sp)
            )
            eval_handles.append(h)
        with torch.no_grad():
            outputs = raw_model.generate(
                **{k: v for k, v in inputs.items() if k != "labels"},
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
            )
        for h in eval_handles:
            h.remove()
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return pred

    def judge_label(pred, image_captions, paired_caption):
        img_strs = _format_captions(_to_caption_strings(image_captions))
        label, _, _ = bert_judge_3way(
            encoder, pred, img_strs, paired_caption,
            margin=judge_margin, threshold=judge_threshold,
            input_type="inconsistent",
        )
        return label

    n_flipped_im2cap = 0
    n_flipped_cap2im = 0
    n_eval_im2cap = 0
    n_eval_cap2im = 0
    by_order = {
        o: {"n_flipped_im2cap": 0, "n_flipped_cap2im": 0, "n_eval_im2cap": 0, "n_eval_cap2im": 0}
        for o in eval_orders
    }

    for i in tqdm(range(len(eval_pairs)), desc="Evaluating"):
        order_ev, data_idx, inputs_clean, inputs_corr, im_span_idcs, cap_span_idcs, sample_im, img_cont, cap_cont = eval_pairs[i]
        img_caps = sample_im["image_captions"]
        paired_cap = sample_im["paired_caption"]

        # Move to GPU only when using this pair
        inputs_clean_gpu = _inputs_to_device(inputs_clean, device)
        inputs_corr_gpu = _inputs_to_device(inputs_corr, device)

        # Base predictions (no intervention)
        with torch.no_grad():
            out_clean = raw_model.generate(**inputs_clean_gpu, do_sample=False, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
            out_corr = raw_model.generate(**inputs_corr_gpu, do_sample=False, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id)
        pred_clean = tokenizer.decode(out_clean[0][inputs_clean_gpu["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred_corr = tokenizer.decode(out_corr[0][inputs_corr_gpu["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        base_clean_label = judge_label(pred_clean, img_caps, paired_cap)
        base_corr_label = judge_label(pred_corr, img_caps, paired_cap)

        if base_clean_label == "image":
            n_eval_im2cap += 1
            by_order[order_ev]["n_eval_im2cap"] += 1
            pred_flip = generate_and_judge(inputs_clean_gpu, im_span_idcs, cap_span_idcs, delta_im_by_layer, delta_cap_by_layer)
            if judge_label(pred_flip, img_caps, paired_cap) == "text":
                n_flipped_im2cap += 1
                by_order[order_ev]["n_flipped_im2cap"] += 1

        if base_corr_label == "text":
            n_eval_cap2im += 1
            by_order[order_ev]["n_eval_cap2im"] += 1
            pred_flip = generate_and_judge(inputs_corr_gpu, im_span_idcs, cap_span_idcs, delta_im_by_layer, delta_cap_by_layer)
            if judge_label(pred_flip, img_caps, paired_cap) == "image":
                n_flipped_cap2im += 1
                by_order[order_ev]["n_flipped_cap2im"] += 1

    flip_rate_im2cap = 100 * n_flipped_im2cap / n_eval_im2cap if n_eval_im2cap > 0 else 0.0
    flip_rate_cap2im = 100 * n_flipped_cap2im / n_eval_cap2im if n_eval_cap2im > 0 else 0.0

    print(f"\nIntervention (layer {layer_idx}), overall:")
    print(f"  im->cap: {n_flipped_im2cap}/{n_eval_im2cap} ({flip_rate_im2cap:.1f}%)")
    print(f"  cap->im: {n_flipped_cap2im}/{n_eval_cap2im} ({flip_rate_cap2im:.1f}%)")
    for o in eval_orders:
        bo = by_order[o]
        ne_i, ne_c = bo["n_eval_im2cap"], bo["n_eval_cap2im"]
        fr_i = 100 * bo["n_flipped_im2cap"] / ne_i if ne_i > 0 else 0.0
        fr_c = 100 * bo["n_flipped_cap2im"] / ne_c if ne_c > 0 else 0.0
        print(f"  order={o}: im->cap {bo['n_flipped_im2cap']}/{ne_i} ({fr_i:.1f}%), cap->im {bo['n_flipped_cap2im']}/{ne_c} ({fr_c:.1f}%)")

    eval_results = {
        "train_orders": train_orders,
        "eval_orders": eval_orders,
        "overall": {
            "im2cap": {"n_flipped": n_flipped_im2cap, "n_eval": n_eval_im2cap, "flip_rate_pct": flip_rate_im2cap},
            "cap2im": {"n_flipped": n_flipped_cap2im, "n_eval": n_eval_cap2im, "flip_rate_pct": flip_rate_cap2im},
        },
        "by_order": {},
    }
    for o in eval_orders:
        bo = by_order[o]
        ne_i, ne_c = bo["n_eval_im2cap"], bo["n_eval_cap2im"]
        eval_results["by_order"][o] = {
            "im2cap": {
                "n_flipped": bo["n_flipped_im2cap"],
                "n_eval": ne_i,
                "flip_rate_pct": 100 * bo["n_flipped_im2cap"] / ne_i if ne_i > 0 else 0.0,
            },
            "cap2im": {
                "n_flipped": bo["n_flipped_cap2im"],
                "n_eval": ne_c,
                "flip_rate_pct": 100 * bo["n_flipped_cap2im"] / ne_c if ne_c > 0 else 0.0,
            },
        }
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval results to {eval_path}")


if __name__ == "__main__":
    main()

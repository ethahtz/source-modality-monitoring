"""
Freeze content, remove: behavioral evaluation with activation patching.

Uses HuggingFace forward hooks (instead of nnsight) to:
1. Cache image_content and cap_content activations from clean inputs at each layer
2. Run generation on "removed" inputs (markers removed) while patching in cached activations

Saves results in the same format as behavioral evaluation (prompt_evaluation.py)
for later metric computation. No selectivity computation.
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from utils import set_seed
from utils.args_utils import get_model_family, get_prompt_template_args, dict_to_object
from utils.data_utils import get_dataset
from utils.prompt_utils import PromptGenerator
from utils.model_utils import load_model_and_preprocess, load_bert_judge
from utils.prompt_evaluation_utils import get_confusion_matrix, process_responses
from utils.parse_spans_utils import get_span
from utils.span_intervention_utils import (
    modify_markers,
    get_deleted_indices_for_remove,
    compute_span_positions_after_remove,
    get_transformer_layers,
    get_num_hidden_layers,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_output_file_path(args):
    """Same path structure as behavioral evaluation, with modification_freeze_content_remove."""
    model_safe = getattr(args, "model_name", "model")
    dataset = getattr(args, "dataset", "unknown")
    if dataset == "mscoco" and getattr(args, "mscoco_split", "val") == "train":
        dataset = "mscoco_train"
    order = getattr(args, "order", "icq")
    input_type = getattr(args, "input_type", "inconsistent")
    modality = getattr(args, "modality_to_report", "image")
    prompt_fmt = getattr(args, "prompt_format", "image_caption")
    seed = getattr(args, "seed", 0)
    fname = f"{prompt_fmt}_{order}_s{seed}.json"
    return os.path.join(
        args.work_dir,
        "results",
        "behavioral_evaluation",
        "modification_freeze_content_remove",
        model_safe,
        dataset,
        input_type,
        modality,
        fname,
    )


def _cache_activations(model, clean_inputs, image_span_positions, caption_span_positions, layers):
    """
    Run forward pass on clean inputs and cache hidden states at image/caption span positions
    for each layer. Returns (image_cache, caption_cache) per layer.
    """
    image_cache = {}
    caption_cache = {}
    handles = []

    def make_capture_hook(layer_idx, im_pos, cap_pos):
        def hook(module, args):
            h = args[0]
            if im_pos:
                image_cache[layer_idx] = h[:, im_pos, :].detach().clone()
            if cap_pos:
                caption_cache[layer_idx] = h[:, cap_pos, :].detach().clone()
            return None

        return hook

    for layer_idx, layer in enumerate(layers):
        h = layer.register_forward_pre_hook(
            make_capture_hook(layer_idx, image_span_positions, caption_span_positions)
        )
        handles.append(h)

    with torch.no_grad():
        model(**clean_inputs)

    for h in handles:
        h.remove()

    return image_cache, caption_cache


def _make_patch_hook(layer_idx, new_im_pos, new_cap_pos, image_cache, caption_cache):
    """Create a pre-hook that patches hidden states with cached activations."""

    def hook(module, args):
        h = args[0].clone()
        if layer_idx in image_cache and new_im_pos:
            for i, pos in enumerate(new_im_pos):
                if pos < h.shape[1] and i < image_cache[layer_idx].shape[1]:
                    h[:, pos, :] = image_cache[layer_idx][:, i, :].to(
                        device=h.device, dtype=h.dtype
                    )
        if layer_idx in caption_cache and new_cap_pos:
            for i, pos in enumerate(new_cap_pos):
                if pos < h.shape[1] and i < caption_cache[layer_idx].shape[1]:
                    h[:, pos, :] = caption_cache[layer_idx][:, i, :].to(
                        device=h.device, dtype=h.dtype
                    )
        return (h,) + args[1:]

    return hook


def get_responses_with_freeze_remove(
    args,
    model,
    processor,
    batch,
    image_cache,
    caption_cache,
    new_image_span_positions,
    new_caption_span_positions,
    layers,
):
    """Generate with frozen activations patched in via forward hooks."""
    handles = []
    for layer_idx, layer in enumerate(layers):
        h = layer.register_forward_pre_hook(
            _make_patch_hook(
                layer_idx,
                new_image_span_positions,
                new_caption_span_positions,
                image_cache,
                caption_cache,
            )
        )
        handles.append(h)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **batch["inputs"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=64,
                min_length=1,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        source_length = batch["inputs"]["input_ids"].size(1)
        responses = process_responses(
            outputs, processor, args, source_length=source_length
        )
    finally:
        for h in handles:
            h.remove()

    return responses


def run_evaluation(args, model, processor, data, bert_encoder=None):
    """
    Run evaluation with freeze_content_remove intervention.
    Processes one sample at a time since span positions differ per sample.
    """
    text_prefix_type = "document" if args.prompt_format == "image_document" else "caption" if args.prompt_format == "image_caption" else "text"
    model_device = next(model.parameters()).device

    try:
        layers = get_transformer_layers(model)
    except AttributeError:
        raise RuntimeError(
            "Could not find transformer layers. Supported: language_model.layers, model.layers, model.model.layers"
        )

    confusion_matrix = defaultdict(int)
    all_predictions = []
    all_model_predictions = []
    all_true_labels = []
    all_judge_scores = []

    # Build list of samples (dataloader with batch_size=1 for per-sample intervention)
    from data.captioning_datasets import LmEvaluationDataCollator

    collator = LmEvaluationDataCollator(
        processor, text_only=(getattr(args, "input_type", "inconsistent") == "text_only")
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating:"):
        inputs = batch["inputs"]
        # Move to device
        inputs = {k: (v.to(model_device) if v is not None else v) for k, v in inputs.items()}

        # Skip if no image or no caption (freeze_content_remove requires both)
        if args.input_type == "image_only" or args.input_type == "text_only":
            # Fall back to normal generation without intervention
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=64,
                    min_length=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            source_length = inputs["input_ids"].size(1)
            responses = process_responses(outputs, processor, args, source_length=source_length)
        else:
            # Get span positions from clean inputs
            image_span_positions = get_span(
                inputs,
                "image_content",
                model_family=args.model_family,
                tokenizer=processor.tokenizer,
                text_prefix_type=text_prefix_type,
            )
            caption_span_positions = get_span(
                inputs,
                "cap_content",
                model_family=args.model_family,
                tokenizer=processor.tokenizer,
                text_prefix_type=text_prefix_type,
            )

            if not image_span_positions or not caption_span_positions:
                # Skip if we can't find both spans (e.g. malformed input)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=64,
                        min_length=1,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                source_length = inputs["input_ids"].size(1)
                responses = process_responses(
                    outputs, processor, args, source_length=source_length
                )
            else:
                # Cache activations from clean inputs
                image_cache, caption_cache = _cache_activations(
                    model, inputs, image_span_positions, caption_span_positions, layers
                )

                # Compute positions after remove
                deleted_sorted = get_deleted_indices_for_remove(
                    inputs,
                    model_family=args.model_family,
                    tokenizer=processor.tokenizer,
                    text_prefix_type=text_prefix_type,
                )
                new_image_pos, new_caption_pos = compute_span_positions_after_remove(
                    image_span_positions, caption_span_positions, deleted_sorted
                )

                # Create removed inputs
                removed_inputs = modify_markers(
                    inputs,
                    model_family=args.model_family,
                    ty="remove",
                    tokenizer=processor.tokenizer,
                    text_prefix_type=text_prefix_type,
                )
                batch["inputs"] = removed_inputs

                responses = get_responses_with_freeze_remove(
                    args,
                    model,
                    processor,
                    batch,
                    image_cache,
                    caption_cache,
                    new_image_pos,
                    new_caption_pos,
                    layers,
                )

        batch_confusion_matrix, predictions, batch_model_preds, batch_true_labels, batch_judge_scores = get_confusion_matrix(
            responses,
            batch["image_captions"],
            batch["paired_captions"],
            args,
            bert_encoder=bert_encoder,
        )

        for k, v in batch_confusion_matrix.items():
            confusion_matrix[k] += v
        all_predictions.extend(predictions)
        all_model_predictions.extend(batch_model_preds)
        all_true_labels.extend(batch_true_labels)
        all_judge_scores.extend(batch_judge_scores)

    sample_input = (
        processor.batch_decode(batch["inputs"]["input_ids"], skip_special_tokens=False)[0]
        if batch["inputs"]["input_ids"].numel() > 0
        else ""
    )
    return confusion_matrix, sample_input, all_predictions, all_model_predictions, all_true_labels, all_judge_scores


def save_result(
    confusion_matrix,
    sample_query,
    args,
    predictions,
    model_predictions,
    true_labels,
    judge_scores,
):
    """Save in same format as prompt_evaluation.save_result."""
    instances = [
        {
            "model_prediction": mp,
            "true_label": tl,
            "judge_score": js,
            "judge_label": jl,
        }
        for mp, tl, js, jl in zip(model_predictions, true_labels, judge_scores, predictions)
    ]
    result = {
        "args": vars(args),
        "sample_query": sample_query,
        "correct": confusion_matrix["n_correct"],
        "misled": confusion_matrix["n_misled"],
        "neither": confusion_matrix["n_neither"],
        "instances": instances,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "instances"}, indent=4))

    output_file = getattr(args, "output_file", None)
    if output_file is not None:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full results saved to {output_file}")


def parse_arguments():


    args = {}

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mscoco", "flickr30k"],
        help="Dataset: mscoco or flickr30k",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Data split (MSCOCO: val/test; Flickr30k: train/val/test). Default: test.",
    )
    parser.add_argument("--version", type=int, required=True)

    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["image_caption", "image_document", "image_text"],
        default="image_caption",
        help="image_caption: Caption: X. image_document: Document: X. image_text: Text: X. (Content from captioning datasets.)",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["consistent", "inconsistent", "text_only", "image_only"],
        default="inconsistent",
        help="consistent: caption from same image; inconsistent: caption from another image; text_only: no image; image_only: no caption.",
    )
    parser.add_argument(
        "--modality_to_report",
        type=str,
        choices=["image", "text"],
        default="image",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
        default="icq",
    )
    parser.add_argument(
        "--is_assistant_prompt",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_pointers",
        type=int,
        default=1,
        help="Use 'Caption:' / 'Document:' prefixes.",
    )

    # Special perturbations (for config compatibility; freeze_content_remove always uses its own intervention)
    parser.add_argument(
        "--modify_inputs",
        type=str,
        choices=["none", "remove", "i2c", "c2i", "swap", "swap_start", "text_substrate"],
        default="none",
    )

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=8)

    # BERT embedding judge
    parser.add_argument(
        "--judge_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="BERT-style sentence encoder for embedding judge.",
    )
    parser.add_argument(
        "--use_bert_judge",
        type=int,
        default=1,
        help="1: use BERT embedding judge; 0: simple string matching.",
    )
    parser.add_argument(
        "--judge_margin",
        type=float,
        default=0.1,
        help="Margin: image if score_A > score_B + margin; text if score_B > score_A + margin.",
    )
    parser.add_argument(
        "--judge_threshold",
        type=float,
        default=0.4,
        help="Minimum score threshold for image/text (neither if both below).",
    )
    parser.add_argument(
        "--inconsistent_sim_threshold",
        type=float,
        default=0.2,
        help="For inconsistent: sampled caption must have max cos_sim < this with image captions.",
    )

    args.update(vars(parser.parse_args()))

    args = dict_to_object(args)
    args.model_family = get_model_family(args)
    args = get_prompt_template_args(args)
    args.output_file = _get_output_file_path(args)

    return args


def main(args):
    set_seed(args.seed)

    model, processor = load_model_and_preprocess(args)

    bert_encoder = None
    if args.use_bert_judge or args.input_type == "inconsistent":
        print(f"Loading sentence encoder: {args.judge_model}")
        cache_dir = f"{args.work_dir}/.cache/huggingface/hub"
        bert_encoder = load_bert_judge(args.judge_model, cache_dir=cache_dir)

    prompt_generator = PromptGenerator(args)
    dataset_encoder = bert_encoder if args.input_type == "inconsistent" else None
    split = getattr(args, "split", "test")
    dataset_test = get_dataset(split, prompt_generator, args, encoder=dataset_encoder)

    confusion_matrix, sample_query, all_predictions, model_predictions, true_labels, judge_scores = run_evaluation(
        args,
        model,
        processor,
        dataset_test,
        bert_encoder=bert_encoder,
    )

    save_result(
        confusion_matrix,
        sample_query,
        args,
        all_predictions,
        model_predictions,
        true_labels,
        judge_scores,
    )


if __name__ == "__main__":
    args = parse_arguments()
    print(args.__dict__)
    main(args)

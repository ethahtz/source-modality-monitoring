"""
Behavioral evaluation for source modality monitoring.

Open-ended generation (no multiple-choice) with BERT embedding judge for 3-way classification:
- correct: prediction matches target modality (image or text)
- misled: prediction matches the other modality
- neither: prediction matches neither

Supports:
- 2 captioning datasets: MSCOCO val (3K val, 2K test) or train (10K/10K/2K), Flickr30k (10K/10K/2K)
- Inconsistent: randomly sample caption from another image (5 captions per image)
- Special perturbations: remove, i2c, c2i, swap, swap_start, text_substrate
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from utils import set_seed
from utils.args_utils import get_model_family, get_prompt_template_args, dict_to_object
from utils.data_utils import get_dataset, get_dataloader
from utils.prompt_utils import PromptGenerator
from utils.model_utils import load_model_and_preprocess, load_bert_judge
from utils.prompt_evaluation_utils import get_responses, get_confusion_matrix
from utils.span_intervention_utils import modify_markers, substitute_image_with_text_substrate

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
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

    # Special perturbations
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

    args = vars(parser.parse_args())

    args = dict_to_object(args)
    args.model_family = get_model_family(args)
    args = get_prompt_template_args(args)
    args.output_file = _get_output_file_path(args)

    return args


def _get_output_file_path(args):
    """Construct hierarchical output path from work_dir and experiment params."""
    model_safe = getattr(args, "model_name", "model")
    dataset = getattr(args, "dataset", "unknown")
    if dataset == "mscoco" and getattr(args, "mscoco_split", "val") == "train":
        dataset = "mscoco_train"
    order = getattr(args, "order", "icq")
    input_type = getattr(args, "input_type", "inconsistent")
    modality = getattr(args, "modality_to_report", "image")
    modify = getattr(args, "modify_inputs", "none")
    prompt_fmt = getattr(args, "prompt_format", "image_caption")
    seed = getattr(args, "seed", 0)
    fname = f"{prompt_fmt}_{order}_s{seed}.json"
    return os.path.join(args.work_dir, "results", "behavioral_evaluation", f"modification_{modify}", model_safe, dataset, input_type, modality, fname)


def run_evaluation(args, model, processor, data, bert_encoder=None):
    loader = get_dataloader(data, processor, args, is_train=False)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating:", unit="its")

    confusion_matrix = defaultdict(int)
    all_predictions = []
    all_model_predictions = []
    all_true_labels = []
    all_judge_scores = []

    text_prefix_type = "document" if args.prompt_format == "image_document" else "caption" if args.prompt_format == "image_caption" else "text"

    for i, batch in pbar:
        if args.modify_inputs != "none":
            if args.modify_inputs == "text_substrate":
                image_labels_list = [
                    captions[0] for captions in batch["image_captions"]
                ]
                batch["inputs"] = substitute_image_with_text_substrate(
                    batch["inputs"],
                    image_labels_list,
                    processor.tokenizer,
                    args.model_family,
                    text_labels=None,
                    text_prefix_type=text_prefix_type,
                )
            else:
                batch["inputs"] = modify_markers(
                    batch["inputs"],
                    model_family=args.model_family,
                    ty=args.modify_inputs,
                    tokenizer=processor.tokenizer,
                    text_prefix_type=text_prefix_type,
                )

        batch["inputs"] = {
            k: (v.to(model.device) if v is not None else v)
            for k, v in batch["inputs"].items()
        }

        responses = get_responses(args, model, processor, batch)

        batch_confusion_matrix, predictions, batch_model_preds, batch_true_labels, batch_judge_scores = get_confusion_matrix(
            responses,
            batch["image_captions"],
            batch["paired_captions"],
            args,
            bert_encoder=bert_encoder,
        )

        for k, v in batch_confusion_matrix.items():
            confusion_matrix[k] += v

        pbar.set_postfix(**confusion_matrix)
        all_predictions.extend(predictions)
        all_model_predictions.extend(batch_model_preds)
        all_true_labels.extend(batch_true_labels)
        all_judge_scores.extend(batch_judge_scores)

    sample_input = processor.batch_decode(batch["inputs"]["input_ids"], skip_special_tokens=False)[0] if batch["inputs"]["input_ids"].numel() > 0 else ""
    return confusion_matrix, sample_input, all_predictions, all_model_predictions, all_true_labels, all_judge_scores


def save_result(confusion_matrix, sample_query, args, input_prompts, predictions, model_predictions, true_labels, judge_scores):
    print("==================== Experiment Results =================")
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
        getattr(dataset_test, "all_sources", []),
        all_predictions,
        model_predictions,
        true_labels,
        judge_scores,
    )


if __name__ == "__main__":
    args = parse_arguments()
    print(args.__dict__)
    main(args)

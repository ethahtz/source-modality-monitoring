"""
Behavioral evaluation with arbitrary labels (e.g. Dax, Wug).

Like prompt_evaluation.py but uses two arbitrary text labels (label_1, label_2)
instead of "image"/"caption". Uses the same BERT embedding judge and evaluation.
"""

import argparse
import json
import os
from collections import defaultdict
from random import choice as rand_choice

import torch
from tqdm import tqdm

from utils import set_seed
from utils.args_utils import get_model_family, get_prompt_template_args, dict_to_object
from utils.data_utils import get_dataset, get_dataloader
from utils.model_utils import load_model_and_preprocess, load_bert_judge
from utils.prompt_evaluation_utils import get_responses, get_confusion_matrix
from utils.span_intervention_utils import remove_vision_markers

device = "cuda" if torch.cuda.is_available() else "cpu"


ARBITRARY_LABEL_MODEL_FAMILIES = ["qwen2.5", "internvl", "gemma"]


def get_arbitrary_label_prompt_template_args(args):
    """Set prompt template args using arbitrary labels (label_1, label_2)."""
    args = get_prompt_template_args(args)

    label_1 = args.label_1
    label_2 = args.label_2

    if args.modality_to_report == "image":
        args.question_template = [f"Report the information associated with label {label_1}.", f"What is the information associated with label {label_1}?"]
    else:
        args.question_template = [f"Report the information associated with label {label_2}.", f"What is the information associated with label {label_2}?"]
    args.answer_template = "Answer: "

    return args


class ArbitraryLabelPromptGenerator:
    """Prompt generator using arbitrary labels (label_1=image, label_2=caption)."""

    def __init__(self, args):
        self.args = args

    def _get_image_part(self):
        if getattr(self.args, "input_type", "inconsistent") == "text_only":
            return ""
        label_1 = self.args.label_1
        if self.args.model_family == "qwen2.5":
            image_content = "<|vision_start|><|image_pad|><|vision_end|>"
        elif self.args.model_family == "internvl":
            image_content = "<IMG_CONTEXT>\n"
        elif self.args.model_family == "gemma":
            image_content = "<start_of_image>\n"
        else:
            raise NotImplementedError(
                f"Arbitrary-label eval only supports {ARBITRARY_LABEL_MODEL_FAMILIES}, got {self.args.model_family}"
            )
        return f"{label_1}: {image_content}. "

    def _get_text_part(self):
        if getattr(self.args, "input_type", "inconsistent") == "image_only":
            return ""
        label_2 = self.args.label_2
        template = rand_choice(self.args.caption_template)
        return f"{label_2}: {template}"

    def _get_question_part(self):
        q = rand_choice(self.args.question_template)
        return f"Question: {q} " + self.args.further_instruction + " "

    def _get_answer_part(self):
        return self.args.answer_template

    def _get_prompt_template(self):
        parts = {
            "image": self._get_image_part(),
            "text": self._get_text_part(),
            "question": self._get_question_part(),
            "answer": self._get_answer_part(),
        }
        order = getattr(self.args, "order", "icq")
        res = []
        for part in order:
            if part == "i":
                res.append(parts["image"])
            elif part == "c":
                res.append(parts["text"])
            elif part == "q":
                res.append(parts["question"])
            else:
                raise ValueError(f"Part {part} not supported.")
        image_text_question = "".join(res).lstrip()

        if getattr(self.args, "is_assistant_prompt", 1):
            if self.args.model_family == "qwen2.5":
                return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n {image_text_question}<|im_end|>\n<|im_start|>assistant\n{parts['answer']}"
            elif self.args.model_family == "internvl":
                return f"<|im_start|>user\n{image_text_question}<|im_end|>\n<|im_start|>assistant\n{parts['answer']}"
            elif self.args.model_family == "gemma":
                return f"<bos><start_of_turn>user\n{image_text_question}<end_of_turn>\n<start_of_turn>model\n{parts['answer']}"
            else:
                raise RuntimeError(
                    f"Arbitrary-label eval only supports {ARBITRARY_LABEL_MODEL_FAMILIES}, got {self.args.model_family}"
                )
        return image_text_question + parts["answer"]

    def __call__(self, format_args=None):
        prompt_template = self._get_prompt_template()
        if format_args is not None:
            return prompt_template.format(*format_args)
        return prompt_template


def _get_output_file_path(args):
    model_safe = getattr(args, "model_name", "model").replace("/", "_")
    dataset = getattr(args, "dataset", "unknown")
    order = getattr(args, "order", "icq")
    input_type = getattr(args, "input_type", "inconsistent")
    modality = getattr(args, "modality_to_report", "image")
    label_1 = getattr(args, "label_1", "Dax")
    label_2 = getattr(args, "label_2", "Wug")
    seed = getattr(args, "seed", 0)
    fname = f"arbitrary_{order}_s{seed}.json"
    return os.path.join(
        args.work_dir,
        "results",
        "behavioral_evaluation_arbitrary_labels",
        model_safe,
        dataset,
        input_type,
        modality,
        f"labels_{label_1}_{label_2}",
        fname,
    )


def run_evaluation(args, model, processor, data, bert_encoder=None):
    """Same evaluation as prompt_evaluation.run_evaluation, but with remove_vision_markers."""
    loader = get_dataloader(data, processor, args, is_train=False)

    confusion_matrix = defaultdict(int)
    all_predictions = []
    all_model_predictions = []
    all_true_labels = []
    all_judge_scores = []

    model_device = next(model.parameters()).device

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating:"):
        batch["inputs"] = remove_vision_markers(batch["inputs"], args.model_family)
        batch["inputs"] = {
            k: (v.to(model_device) if v is not None else v)
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


def save_result(confusion_matrix, sample_query, args, predictions, model_predictions, true_labels, judge_scores):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mscoco", "flickr30k"],
        help="Captioning dataset: mscoco or flickr30k",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["image_caption", "image_document"],
        default="image_caption",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["consistent", "inconsistent", "text_only", "image_only"],
        default="inconsistent",
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
    parser.add_argument("--is_assistant_prompt", type=int, default=1)
    parser.add_argument("--use_pointers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument(
        "--label_1",
        type=str,
        default="Dax",
        help="Arbitrary label for image modality.",
    )
    parser.add_argument(
        "--label_2",
        type=str,
        default="Wug",
        help="Arbitrary label for caption modality.",
    )
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
    parser.add_argument("--inconsistent_sim_threshold", type=float, default=0.2)

    args = vars(parser.parse_args())

    args = dict_to_object(args)
    args.model_family = get_model_family(args)
    if args.model_family not in ARBITRARY_LABEL_MODEL_FAMILIES:
        raise ValueError(
            f"Arbitrary-label eval only supports {ARBITRARY_LABEL_MODEL_FAMILIES}, got {args.model_family}"
        )
    args = get_arbitrary_label_prompt_template_args(args)
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

    prompt_generator = ArbitraryLabelPromptGenerator(args)
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

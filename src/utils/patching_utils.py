"""Dataset and input preparation for transformation vector training (captioning datasets)."""

import os
import random

import torch

from utils.prompt_utils import PromptGenerator
from utils.args_utils import get_prompt_template_args
from data.captioning_datasets import MultimodalCaptioningDataset
from utils.prompt_evaluation_utils import get_prompt, get_paired_caption, _to_caption_strings
from data.captioning_datasets import _format_caption, _format_captions


def get_prompt_args(model_name, dataset, order="icq", work_dir=None, modality_to_report="image", **kwargs):
    """Build args object for prompt generation (captioning datasets)."""
    from utils.args_utils import dict_to_object, get_model_family

    work_dir = work_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = {
        "model_name": model_name,
        "dataset": dataset,
        "input_type": kwargs.get("input_type", "inconsistent"),
        "modality_to_report": modality_to_report,
        "order": order,
        "use_pointers": 1,
        "prompt_format": kwargs.get("prompt_format", "image_caption"),
        "work_dir": work_dir,
        "seed": kwargs.get("seed", 42),
        "mscoco_split": kwargs.get("mscoco_split", "val"),
    }
    args = dict_to_object(args)
    args.model_family = get_model_family(args)
    args = get_prompt_template_args(args)
    return args


def get_dataset_with_target_modality_minimal_pairs(args, target_modality, split="train"):
    """
    Get captioning dataset with modality_to_report set.
    target_modality: 'image' or 'text'.
    Returns MultimodalCaptioningDataset where each sample has prompt asking about that modality.
    """
    assert target_modality in ["image", "text"]
    args.modality_to_report = target_modality
    get_prompt_template_args(args)  # Update question/answer templates for target modality
    prompt_generator = PromptGenerator(args)
    from utils.data_utils import get_dataset

    return get_dataset(split, prompt_generator, args)


def prepare_model_inputs(datapoint, processor, device="cuda"):
    """Prepare model inputs from a datapoint (source, image, etc.)."""
    image = datapoint["image"]
    if isinstance(image, list):
        image = image[0]
    model_inputs = processor(text=datapoint["source"], images=image, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def prepare_model_inputs_with_continuation(datapoint, processor, continuation, device="cuda"):
    """
    Build inputs with prompt + continuation for generation loss.
    continuation: string to append (e.g. paired_caption or image_caption).
    Returns inputs dict, labels (shifted, -100 for prompt), and prompt_len.
    """
    prompt_text = datapoint["source"] + " "
    full_text = prompt_text + continuation
    image = datapoint["image"]
    if isinstance(image, list):
        image = image[0]

    # Get prompt-only length by processing prompt_text + image (same structure as full)
    inputs_prompt = processor(text=prompt_text, images=image, return_tensors="pt")
    prompt_len = inputs_prompt["input_ids"].shape[1]

    model_inputs = processor(text=full_text, images=image, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # Labels: labels[i] = input_ids[i+1] (next-token prediction), mask prompt
    input_ids = model_inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:].clone()
    labels[:, :prompt_len] = -100
    model_inputs["labels"] = labels

    return model_inputs, prompt_len

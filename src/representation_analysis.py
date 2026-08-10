"""
Representation analysis (cosine similarity, PCA/t-SNE, linear probe) for captioning datasets.

Uses this repo's MSCOCO/Flickr30k pipeline: get_prompt_args, MultimodalCaptioningDataset via
get_dataset_with_target_modality_minimal_pairs, and tokenizer-aware get_span.

Span extraction (utils.parse_spans_utils) supports model_family: qwen2.5, internvl, gemma.
"""
import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, os.path.join(_project_root, "src"))

from utils.args_utils import get_model_family
from utils.model_utils import load_model_and_preprocess
from utils.parse_spans_utils import get_span
from utils.patching_utils import (
    get_dataset_with_target_modality_minimal_pairs,
    get_prompt_args,
    prepare_model_inputs,
)
from utils.span_intervention_utils import get_num_hidden_layers

# Span backends in parse_spans_utils (image wrapper tokens + Caption:/Document: + dot)
SPAN_SUPPORTED_MODEL_FAMILIES = frozenset({"qwen2.5", "internvl", "gemma"})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _span_names(span_type):
    """Map CLI span_type to get_span keys (image_content, cap_content, ...)."""
    if span_type not in ("content", "start", "end"):
        raise ValueError(f"span_type must be content|start|end, got {span_type}")
    return f"image_{span_type}", f"cap_{span_type}"


def collect_representations(
    model,
    processor,
    image_dataset,
    span_type,
    model_family,
    text_prefix_type,
    num_samples=100,
    seed=44,
    layer_idx=0,
):
    """
    Collect one image-token and one caption-token representation per sampled example.
    """
    set_seed(seed)
    im_span_name, cap_span_name = _span_names(span_type)
    tokenizer = processor.tokenizer

    dataset_size = len(image_dataset)
    if dataset_size < num_samples:
        raise ValueError(f"Dataset has {dataset_size} examples; need at least {num_samples}")

    sampled_indices = random.sample(range(dataset_size), num_samples)

    num_layers = get_num_hidden_layers(model)
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx must be in [0, {num_layers - 1}], got {layer_idx}")

    device = next(model.parameters()).device

    image_reprs = []
    caption_reprs = []

    print(f"Collecting {num_samples} representations per modality (1 token each per sample)...")
    print(f"Using layer {layer_idx} of {num_layers}; span names: {im_span_name}, {cap_span_name}")

    for idx in tqdm(sampled_indices, desc="Processing examples"):
        datapoint = image_dataset[idx]
        inputs = prepare_model_inputs(datapoint, processor, device=device)

        im_span_idcs = get_span(
            inputs,
            im_span_name,
            model_family=model_family,
            tokenizer=tokenizer,
            text_prefix_type=text_prefix_type,
        )
        cap_span_idcs = get_span(
            inputs,
            cap_span_name,
            model_family=model_family,
            tokenizer=tokenizer,
            text_prefix_type=text_prefix_type,
        )

        if len(im_span_idcs) == 0 or len(cap_span_idcs) == 0:
            continue

        model_to_use = model.model if hasattr(model, "model") else model
        with torch.no_grad():
            outputs = model_to_use(**inputs, output_hidden_states=True)
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
            elif isinstance(outputs, dict) and "hidden_states" in outputs:
                hidden_states = outputs["hidden_states"]
            elif hasattr(outputs, "language_model_outputs"):
                hidden_states = outputs.language_model_outputs.hidden_states
            else:
                raise ValueError("Could not find hidden_states in model output")

            layer_hidden_state = hidden_states[layer_idx].squeeze().cpu()

        im_acts = layer_hidden_state[im_span_idcs]
        sampled_im_idx = random.choice(range(im_acts.shape[0]))
        im_repr = im_acts[sampled_im_idx : sampled_im_idx + 1]
        image_reprs.append(im_repr)

        cap_acts = layer_hidden_state[cap_span_idcs]
        sampled_cap_idx = random.choice(range(cap_acts.shape[0]))
        cap_repr = cap_acts[sampled_cap_idx : sampled_cap_idx + 1]
        caption_reprs.append(cap_repr)

    if len(image_reprs) < num_samples or len(caption_reprs) < num_samples:
        raise ValueError(
            f"Not enough valid spans: image={len(image_reprs)}, caption={len(caption_reprs)} "
            f"(need {num_samples}). Try more data, another span_type, or check tokenizer/prompt."
        )

    image_reprs_stacked = torch.cat(image_reprs, dim=0)[:num_samples]
    caption_reprs_stacked = torch.cat(caption_reprs, dim=0)[:num_samples]
    hidden_dim = image_reprs_stacked.shape[-1]

    print(
        f"Collected {image_reprs_stacked.shape[0]} image and "
        f"{caption_reprs_stacked.shape[0]} caption vectors; hidden_dim={hidden_dim}"
    )

    return {
        "image_reprs": image_reprs_stacked,
        "caption_reprs": caption_reprs_stacked,
        "layer_idx": layer_idx,
        "hidden_dim": hidden_dim,
    }


def compute_cosine_similarities(image_reprs, caption_reprs):
    image_reprs = image_reprs.cpu()
    caption_reprs = caption_reprs.cpu()

    n_im = image_reprs.shape[0]
    im_im_cos_sims = []
    for i in range(n_im):
        for j in range(i + 1, n_im):
            im_im_cos_sims.append(
                cosine_similarity(image_reprs[i : i + 1], image_reprs[j : j + 1], dim=1).item()
            )
    within_image = np.mean(im_im_cos_sims) if im_im_cos_sims else None

    n_cap = caption_reprs.shape[0]
    cap_cap_cos_sims = []
    for i in range(n_cap):
        for j in range(i + 1, n_cap):
            cap_cap_cos_sims.append(
                cosine_similarity(
                    caption_reprs[i : i + 1], caption_reprs[j : j + 1], dim=1
                ).item()
            )
    within_caption = np.mean(cap_cap_cos_sims) if cap_cap_cos_sims else None

    im_cap_cos_sims = []
    for i in range(n_im):
        for j in range(n_cap):
            im_cap_cos_sims.append(
                cosine_similarity(
                    image_reprs[i : i + 1], caption_reprs[j : j + 1], dim=1
                ).item()
            )
    across_modality = np.mean(im_cap_cos_sims) if im_cap_cos_sims else None

    return {
        "within_image": within_image,
        "within_caption": within_caption,
        "across_modality": across_modality,
    }


def visualize_pca_tsne(image_reprs, caption_reprs, output_dir, span_type, model_name, dataset):
    all_reprs = torch.cat([image_reprs, caption_reprs], dim=0).float().cpu().numpy()
    labels = np.array([0] * image_reprs.shape[0] + [1] * caption_reprs.shape[0])

    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=44)
    pca_reprs = pca.fit_transform(all_reprs)

    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=44)
    tsne_reprs = tsne.fit_transform(all_reprs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    im_mask = labels == 0
    cap_mask = labels == 1

    ax1.scatter(
        pca_reprs[im_mask, 0],
        pca_reprs[im_mask, 1],
        c="blue",
        label="Image",
        alpha=0.6,
        s=30,
    )
    ax1.scatter(
        pca_reprs[cap_mask, 0],
        pca_reprs[cap_mask, 1],
        c="red",
        label="Caption",
        alpha=0.6,
        s=30,
    )
    ax1.set_xlabel("PC1", fontsize=12)
    ax1.set_ylabel("PC2", fontsize=12)
    ax1.set_title(
        f"PCA (explained variance: {pca.explained_variance_ratio_.sum():.2%})",
        fontsize=14,
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(
        tsne_reprs[im_mask, 0],
        tsne_reprs[im_mask, 1],
        c="blue",
        label="Image",
        alpha=0.6,
        s=30,
    )
    ax2.scatter(
        tsne_reprs[cap_mask, 0],
        tsne_reprs[cap_mask, 1],
        c="red",
        label="Caption",
        alpha=0.6,
        s=30,
    )
    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)
    ax2.set_title("t-SNE", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_model = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"pca_tsne_{span_type}_{safe_model}_{dataset}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved PCA/t-SNE plot to {save_path}")
    plt.close()

    return pca_reprs, tsne_reprs


def _linear_probe_cv_single(all_reprs, y, n_folds, task_name):
    """
    Stratified K-fold logistic regression; same reporting for true labels or control labels.
    y must be length n_samples with two classes (0 = image, 1 = caption).
    """
    linear_probe = LogisticRegression(random_state=44, max_iter=1000)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=44)

    print(f"Linear probe ({task_name}): {n_folds}-fold stratified CV...")
    cv_accuracy = cross_val_score(linear_probe, all_reprs, y, cv=skf, scoring="accuracy")
    cv_precision = cross_val_score(linear_probe, all_reprs, y, cv=skf, scoring="precision")
    cv_recall = cross_val_score(linear_probe, all_reprs, y, cv=skf, scoring="recall")
    cv_f1 = cross_val_score(linear_probe, all_reprs, y, cv=skf, scoring="f1")

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_reprs, y)):
        X_train, X_val = all_reprs[train_idx], all_reprs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        probe = LogisticRegression(random_state=44, max_iter=1000)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_val)
        fold_scores.append(
            {
                "fold": fold_idx + 1,
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            }
        )

    print(f"  Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  Recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    return {
        "cv_accuracy": float(cv_accuracy.mean()),
        "cv_accuracy_std": float(cv_accuracy.std()),
        "cv_precision": float(cv_precision.mean()),
        "cv_precision_std": float(cv_precision.std()),
        "cv_recall": float(cv_recall.mean()),
        "cv_recall_std": float(cv_recall.std()),
        "cv_f1": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "fold_scores": fold_scores,
    }


def compute_linear_probe_cv(image_reprs, caption_reprs, n_folds=3, control_shuffle_seed=44):
    """
    True-label probe plus control: same CV with labels randomly permuted (no image/caption signal).
    control_shuffle_seed fixes the permutation for reproducibility.
    """
    all_reprs = torch.cat([image_reprs, caption_reprs], dim=0).float().cpu().numpy()
    num_image_samples = image_reprs.shape[0]
    true_labels = np.array([0] * num_image_samples + [1] * caption_reprs.shape[0])

    results_true = _linear_probe_cv_single(
        all_reprs, true_labels, n_folds, task_name="true labels"
    )

    rng = np.random.RandomState(control_shuffle_seed)
    shuffled_labels = rng.permutation(true_labels)
    results_control = _linear_probe_cv_single(
        all_reprs, shuffled_labels, n_folds, task_name="control (shuffled labels)"
    )

    return {
        "true_labels": results_true,
        "control_shuffled_labels": results_control,
        "control_shuffle_seed": int(control_shuffle_seed),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Representation analysis for MSCOCO/Flickr30k captioning setup"
    )
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["mscoco", "flickr30k"])
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Captioning split (sizes defined in data/captioning_datasets.py).",
    )
    parser.add_argument(
        "--mscoco_split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="MSCOCO only: val2017 annotations vs train2017.",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="image_caption",
        choices=["image_caption", "image_document"],
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="inconsistent",
        help="Same as behavioral eval: paired caption from same vs other image.",
    )
    parser.add_argument(
        "--modality_to_report",
        type=str,
        default="image",
        choices=["image", "text"],
        help="Which task modality the prompt targets (updates question template).",
    )
    parser.add_argument(
        "--span_type",
        type=str,
        choices=["content", "start", "end"],
        default="content",
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--input_order",
        type=str,
        default="icq",
        choices=["icq", "iqc", "qic", "qci", "cqi", "ciq"],
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Hidden layer index for activations (0 = early layer).",
    )
    parser.add_argument(
        "--control_shuffle_seed",
        type=int,
        default=None,
        help="RNG seed for permuting labels in the linear-probe control task (default: --seed).",
    )
    return parser.parse_args()



def main():
    args = parse_arguments()
    set_seed(args.seed)

    print("=" * 80)
    print("Representation analysis (captioning datasets)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} split={args.split} mscoco_split={args.mscoco_split}")
    print(f"prompt_format={args.prompt_format} input_type={args.input_type} order={args.input_order}")
    print(f"span_type={args.span_type} num_samples={args.num_samples} layer_idx={args.layer_idx}")
    print("=" * 80)

    pmpt_args = get_prompt_args(
        args.model_name,
        args.dataset,
        order=args.input_order,
        work_dir=args.work_dir,
        modality_to_report=args.modality_to_report,
        seed=args.seed,
        mscoco_split=args.mscoco_split,
        prompt_format=args.prompt_format,
        input_type=args.input_type,
    )
    model_family = get_model_family(pmpt_args)
    pmpt_args.model_family = model_family

    if model_family not in SPAN_SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Span extraction in parse_spans_utils supports {sorted(SPAN_SUPPORTED_MODEL_FAMILIES)}; "
            f"this checkpoint resolves to model_family={model_family}."
        )

    text_prefix_type = "document" if args.prompt_format == "image_document" else "caption"

    output_dir = os.path.join(
        args.work_dir,
        "results",
        "representation_analysis",
        args.model_name.replace("/", "_"),
        args.dataset,
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading model...")
    model, processor = load_model_and_preprocess(pmpt_args)

    print("Loading captioning dataset (image-modality prompt)...")
    train_image_dataset = get_dataset_with_target_modality_minimal_pairs(
        pmpt_args, "image", split=args.split
    )
    print(f"Dataset size: {len(train_image_dataset)}")

    reprs_dict = collect_representations(
        model,
        processor,
        train_image_dataset,
        args.span_type,
        model_family,
        text_prefix_type,
        num_samples=args.num_samples,
        seed=args.seed,
        layer_idx=args.layer_idx,
    )
    image_reprs = reprs_dict["image_reprs"]
    caption_reprs = reprs_dict["caption_reprs"]
    layer_idx = reprs_dict["layer_idx"]

    print("\nCosine similarities...")
    cos_sim_results = compute_cosine_similarities(image_reprs, caption_reprs)
    print(f"  Within-image: {cos_sim_results['within_image']:.4f}")
    print(f"  Within-caption: {cos_sim_results['within_caption']:.4f}")
    print(f"  Across modality: {cos_sim_results['across_modality']:.4f}")

    print("\nPCA / t-SNE...")
    visualize_pca_tsne(
        image_reprs,
        caption_reprs,
        output_dir,
        args.span_type,
        args.model_name,
        args.dataset,
    )

    control_seed = (
        args.control_shuffle_seed if args.control_shuffle_seed is not None else args.seed
    )
    print("\nLinear probe (true labels + control: shuffled labels)...")
    linear_probe_bundle = compute_linear_probe_cv(
        image_reprs, caption_reprs, n_folds=3, control_shuffle_seed=control_seed
    )

    results = {
        "cosine_similarities": {
            "within_image": float(cos_sim_results["within_image"]),
            "within_caption": float(cos_sim_results["within_caption"]),
            "across_modality": float(cos_sim_results["across_modality"]),
        },
        "linear_probe_metrics": {
            "true_labels": linear_probe_bundle["true_labels"],
            "control_shuffled_labels": linear_probe_bundle["control_shuffled_labels"],
            "control_shuffle_seed": linear_probe_bundle["control_shuffle_seed"],
        },
        "layer_idx": int(layer_idx),
        "num_samples": args.num_samples,
        "span_type": args.span_type,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "split": args.split,
        "mscoco_split": args.mscoco_split,
        "prompt_format": args.prompt_format,
        "input_type": args.input_type,
        "input_order": args.input_order,
        "modality_to_report": args.modality_to_report,
    }

    results_path = os.path.join(output_dir, f"results_{args.span_type}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")

    lt = linear_probe_bundle["true_labels"]
    lc = linear_probe_bundle["control_shuffled_labels"]
    print("\nLinear probe summary (true vs control shuffled labels):")
    print(
        f"  Accuracy: {lt['cv_accuracy']:.4f} ± {lt['cv_accuracy_std']:.4f}  vs  "
        f"{lc['cv_accuracy']:.4f} ± {lc['cv_accuracy_std']:.4f}"
    )
    print(
        f"  F1:       {lt['cv_f1']:.4f} ± {lt['cv_f1_std']:.4f}  vs  "
        f"{lc['cv_f1']:.4f} ± {lc['cv_f1_std']:.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Open-ended generation and BERT embedding judge for 3-way classification (image / text / neither)."""

import random
from typing import List, Optional

import numpy as np
import torch

# --- Captioning datasets (MSCOCO, Flickr30k) ---


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _to_caption_strings(captions):
    """Normalize captions to list of strings (handles MSCOCO dict format and list of strings)."""
    if captions is None:
        return []
    if isinstance(captions, str):
        return [captions]
    out = []
    for c in captions:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict) and "caption" in c:
            out.append(c["caption"])
        else:
            out.append(str(c))
    return out


def get_paired_caption(args, captions, ex_id, all_captions_by_image, encoder=None):
    """
    For captioning: pick 1 caption to pair with the image.
    - consistent: random from same image's 5 captions
    - inconsistent: random from another image's 5 captions, with cos_sim < threshold to image captions
    - image_only: no caption (None)
    - text_only: caption for text-only prompt (no image)
    """
    if args.input_type == "consistent":
        strs = _to_caption_strings(captions)
        return random.choice(strs) if strs else ""
    elif args.input_type == "inconsistent":
        img_caption_strs = _to_caption_strings(captions)
        sim_threshold = getattr(args, "inconsistent_sim_threshold", 0.2)
        n = len(all_captions_by_image)
        other_indices = [j for j in range(n) if j != ex_id]
        random.shuffle(other_indices)

        if encoder is not None and img_caption_strs:
            img_embs = encoder.encode(img_caption_strs, convert_to_numpy=True)
            candidates = []
            for other_idx in other_indices:
                other_captions = all_captions_by_image[other_idx]
                other_strs = _to_caption_strings(other_captions)
                for cand in other_strs:
                    cand_emb = encoder.encode([cand], convert_to_numpy=True)[0]
                    max_sim = max(_cosine_sim(cand_emb, ae) for ae in img_embs)
                    if max_sim < sim_threshold:
                        return cand
                    candidates.append((cand, max_sim))
            if candidates:
                return min(candidates, key=lambda x: x[1])[0]

        other_idx = random.choice(other_indices)
        other_strs = _to_caption_strings(all_captions_by_image[other_idx])
        return random.choice(other_strs) if other_strs else ""
    elif args.input_type == "image_only":
        return None
    elif args.input_type == "text_only":
        strs = _to_caption_strings(captions)
        return random.choice(strs) if strs else ""
    strs = _to_caption_strings(captions)
    return random.choice(strs) if strs else ""


def get_prompt(args, prompt_generator, paired_caption):
    """Build prompt for captioning datasets. paired_caption is the string to show."""
    if paired_caption is None:
        return prompt_generator(None)
    return prompt_generator([paired_caption])


def get_responses(args, model, processor, batch):
    """Get open-ended model responses (no multiple-choice)."""
    with torch.no_grad():
        outputs = model.generate(
            **batch["inputs"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            min_length=1,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    responses = process_responses(
        outputs, processor, args, source_length=batch["inputs"]["input_ids"].size(1)
    )
    return responses


def process_responses(outputs, processor, args, source_length):
    """Extract and clean model responses."""
    if hasattr(outputs, "choices"):
        responses = [outputs.choices[0].message.content]
    else:
        responses = processor.batch_decode(
            outputs[:, source_length:], skip_special_tokens=True
        )
    responses = [t.strip().strip(".").strip("'").strip("*").lower() for t in responses]
    return responses


def bert_judge_3way(
    encoder,
    prediction: str,
    image_captions: List[str],
    paired_caption: Optional[str],
    margin: float = 0.0,
    threshold: float = 0.0,
    input_type: str = "inconsistent",
    modality_to_report: str = "image",
):
    """
    BERT embedding-based 3-way classification.
    Returns (judgment, score_a, score_b).

    When input_type is consistent or there's only one input (image_only/text_only),
    use a single score vs threshold instead of comparing two scores.
    """
    pred_emb = encoder.encode([prediction], convert_to_numpy=True)[0]

    # Compute score_a (max sim to image captions)
    if image_captions:
        a_embs = encoder.encode(image_captions, convert_to_numpy=True)
        sims_a = [_cosine_sim(pred_emb, ae) for ae in a_embs]
        score_a = max(sims_a)
    else:
        score_a = 0.0

    # Compute score_b (sim to paired caption)
    if paired_caption is not None:
        b_emb = encoder.encode([paired_caption], convert_to_numpy=True)[0]
        score_b = _cosine_sim(pred_emb, b_emb)
    else:
        score_b = 0.0

    # Single-input or consistent: compare one score to threshold
    if input_type == "image_only":
        return ("image" if score_a > threshold else "neither", score_a, score_b)
    if input_type == "text_only":
        return ("text" if score_b > threshold else "neither", score_a, score_b)
    if input_type == "consistent":
        score = score_a if modality_to_report == "image" else score_b
        modality = "image" if modality_to_report == "image" else "text"
        return (modality if score > threshold else "neither", score_a, score_b)

    # Inconsistent: compare score_a vs score_b with margin
    if score_a > score_b + margin and score_a > threshold:
        return ("image", score_a, score_b)
    if score_b > score_a + margin and score_b > threshold:
        return ("text", score_a, score_b)
    return ("neither", score_a, score_b)


def get_confusion_matrix(
    responses,
    image_captions_batch,
    paired_captions_batch,
    args,
    bert_encoder=None,
):
    """
    Compute confusion matrix for captioning datasets.
    Returns (confusion_dict, all_predictions, model_predictions, true_labels, judge_scores).
    """
    n_correct, n_misled, n_neither = 0, 0, 0
    all_predictions = []
    model_predictions = []
    true_labels = []
    judge_scores = []

    margin = getattr(args, "judge_margin", 0.0)
    threshold = getattr(args, "judge_threshold", 0.0)

    for i, response in enumerate(responses):
        image_captions = image_captions_batch[i]
        paired_caption = paired_captions_batch[i]

        model_predictions.append(response)
        true_labels.append({
            "image_captions": image_captions,
            "paired_caption": paired_caption,
        })

        if bert_encoder is not None:
            judgment, score_a, score_b = bert_judge_3way(
                bert_encoder,
                response,
                image_captions,
                paired_caption,
                margin=margin,
                threshold=threshold,
                input_type=getattr(args, "input_type", "inconsistent"),
                modality_to_report=getattr(args, "modality_to_report", "image"),
            )
            judge_scores.append({"score_a": float(score_a), "score_b": float(score_b)})
            if judgment == "image":
                if args.modality_to_report == "image":
                    n_correct += 1
                    all_predictions.append("correct")
                else:
                    n_misled += 1
                    all_predictions.append("misled")
            elif judgment == "text":
                if args.modality_to_report == "text":
                    n_correct += 1
                    all_predictions.append("correct")
                else:
                    n_misled += 1
                    all_predictions.append("misled")
            else:
                n_neither += 1
                all_predictions.append("neither")
        else:
            judge_scores.append({"score_a": None, "score_b": None})
            response_clean = response.lower().strip()
            matches_image = any(
                cap.lower() in response_clean or response_clean in cap.lower()
                for cap in image_captions
            )
            matches_text = (
                paired_caption is not None
                and (
                    paired_caption.lower() in response_clean
                    or response_clean in paired_caption.lower()
                )
            )
            if args.modality_to_report == "image":
                if matches_image and not matches_text:
                    n_correct += 1
                    all_predictions.append("correct")
                elif matches_text:
                    n_misled += 1
                    all_predictions.append("misled")
                else:
                    n_neither += 1
                    all_predictions.append("neither")
            else:
                if matches_text and not matches_image:
                    n_correct += 1
                    all_predictions.append("correct")
                elif matches_image:
                    n_misled += 1
                    all_predictions.append("misled")
                else:
                    n_neither += 1
                    all_predictions.append("neither")

    return {
        "n_correct": n_correct,
        "n_misled": n_misled,
        "n_neither": n_neither,
    }, all_predictions, model_predictions, true_labels, judge_scores

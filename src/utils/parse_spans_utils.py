"""Span extraction for image and text (caption/document) in model inputs."""

import torch

SPECIAL_TOKENS = {
    "qwen2.5": {
        "vision_start": 151652,
        "vision_end": 151653,
        "image_pad": 151655,
        "dot": 13,
    },
    "internvl": {
        "img_start": 151665,
        "img_end": 151666,
        "img_context": 151667,
        "dot": 13,
    },
    "gemma": {
        "boi": 255999,
        "eoi": 256000,
        "image_token": 262144,
        "dot": 236761,
    },
}


def get_text_prefix_from_tokenizer(tokenizer, prefix_type="caption"):
    """
    Get token IDs for text prefix (Caption: or Document:) from tokenizer.
    Returns list of tensors: [with_prepending_space, without_prepending_space].
    Caller should try _find_subseq with each until one matches.
    """
    if prefix_type == "document":
        label = "Document:"
    else:
        label = "Caption:"
    with_space = torch.tensor(tokenizer.encode(" " + label, add_special_tokens=False))
    without_space = torch.tensor(tokenizer.encode(label, add_special_tokens=False))
    return [with_space, without_space]


def _find_subseq(haystack: torch.Tensor, needle: torch.Tensor, start: int = 0):
    """Return first start index of needle in haystack (1D tensors), else None."""
    if needle.numel() == 0:
        return None
    for i in range(start, haystack.numel() - needle.numel() + 1):
        if torch.equal(haystack[i : i + needle.numel()], needle):
            return i
    return None


def _first_token_after(haystack: torch.Tensor, token_id: int, start: int):
    """Return index of first occurrence of token_id at/after start, else None."""
    idxs = torch.where(haystack[start:] == token_id)[0]
    return (start + idxs[0].item()) if idxs.numel() else None


def get_span(inputs, span_type, model_family="qwen2.5", tokenizer=None, text_prefix_type="caption"):
    """
    Get span indices for image or text (caption/document).
    For text span: tokenizer and text_prefix_type ('caption' or 'document') required.
    """
    input_ids = inputs["input_ids"][0].detach().cpu()
    tok = SPECIAL_TOKENS[model_family]

    # Image span
    if model_family == "internvl":
        img_start_id = tok["img_start"]
        img_end_id = tok["img_end"]
    elif model_family == "qwen2.5":
        img_start_id = tok["vision_start"]
        img_end_id = tok["vision_end"]
    elif model_family == "gemma":
        img_start_id = tok["boi"]
        img_end_id = tok["eoi"]
    else:
        raise ValueError(f"Unknown model_family={model_family}")

    s = torch.where(input_ids == img_start_id)[0]
    e = torch.where(input_ids == img_end_id)[0]
    img_start_idx = s[0].item() if s.numel() else None
    img_end_idx = e[e > img_start_idx][0].item() if (s.numel() and e.numel()) else None

    if img_start_idx is not None and img_end_idx is not None:
        img_content_idcs = list(range(img_start_idx + 1, img_end_idx))
        image_spans = {
            "image_all": list(range(img_start_idx, img_end_idx + 1)),
            "image_start": [img_start_idx],
            "image_end": [img_end_idx],
            "image_content": img_content_idcs,
        }
    else:
        image_spans = {
            "image_all": [],
            "image_start": [],
            "image_end": [],
            "image_content": [],
        }

    # Text span (caption or document)
    cap_start_idx = None
    actual_cap_len = 0
    if tokenizer is not None:
        text_prefix_options = get_text_prefix_from_tokenizer(tokenizer, text_prefix_type)
        for prefix_tensor in text_prefix_options:
            prefix_cpu = prefix_tensor.detach().cpu() if hasattr(prefix_tensor, "detach") else prefix_tensor
            cap_start_idx = _find_subseq(input_ids, prefix_cpu, start=0)
            if cap_start_idx is not None:
                actual_cap_len = prefix_tensor.numel() if hasattr(prefix_tensor, "numel") else len(prefix_tensor)
                break

    cap_end_idx = None
    if cap_start_idx is not None:
        cap_content_start = int(cap_start_idx) + actual_cap_len
        cap_end_idx = _first_token_after(input_ids, tok["dot"], start=cap_content_start)

    if cap_start_idx is not None and cap_end_idx is not None:
        caption_spans = {
            "cap_all": list(range(int(cap_start_idx), int(cap_end_idx) + 1)),
            "cap_start": list(range(int(cap_start_idx), int(cap_start_idx) + actual_cap_len)),
            "cap_content": list(range(int(cap_start_idx) + actual_cap_len, int(cap_end_idx))),
            "cap_end": [int(cap_end_idx)],
        }
    else:
        caption_spans = {
            "cap_all": [],
            "cap_start": [],
            "cap_content": [],
            "cap_end": [],
        }

    all_spans = {**image_spans, **caption_spans}
    return all_spans.get(span_type, [])

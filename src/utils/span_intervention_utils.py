"""Special perturbations: modify markers (remove, i2c, c2i, swap, swap_start) and text substrate."""

import torch

from utils.parse_spans_utils import (
    SPECIAL_TOKENS,
    _find_subseq,
    _first_token_after,
    get_text_prefix_from_tokenizer,
)


def remove_vision_markers(inputs, model_family):
    """Remove image start/end delimiter tokens from tokenized inputs.

    After the processor embeds image features at <image_pad> positions,
    the start/end delimiter tokens (e.g. <|vision_start|> / <|vision_end|>)
    are no longer needed.  Removing them leaves the arbitrary text labels
    as the sole delimiters around image content.

    For model families without separate start/end markers (e.g. LLaVA),
    this is a no-op.
    """
    tokens_to_remove = set()
    if model_family in ["qwen2.5", "qwen3"]:
        tok = SPECIAL_TOKENS.get("qwen2.5")
        if tok:
            tokens_to_remove = {int(tok["vision_start"]), int(tok["vision_end"])}
    elif model_family == "internvl":
        tok = SPECIAL_TOKENS["internvl"]
        tokens_to_remove = {int(tok["img_start"]), int(tok["img_end"])}
    elif model_family == "gemma":
        tok = SPECIAL_TOKENS["gemma"]
        tokens_to_remove = {int(tok["boi"]), int(tok["eoi"])}

    if not tokens_to_remove:
        return inputs

    out = {k: v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    device = input_ids.device
    B, L = input_ids.shape

    keep_masks = []
    for b in range(B):
        mask = torch.ones(L, dtype=torch.bool, device=device)
        for token_id in tokens_to_remove:
            mask &= (input_ids[b] != token_id)
        keep_masks.append(mask)

    new_rows = [input_ids[b][keep_masks[b]] for b in range(B)]
    out["input_ids"] = torch.stack(new_rows, dim=0)

    if "attention_mask" in inputs:
        out["attention_mask"] = torch.stack(
            [inputs["attention_mask"][b][keep_masks[b]] for b in range(B)], dim=0
        )

    if "token_type_ids" in inputs:
        out["token_type_ids"] = torch.stack(
            [inputs["token_type_ids"][b][keep_masks[b]] for b in range(B)], dim=0
        )

    return out


def modify_markers(inputs, model_family="qwen2.5", ty="swap", tokenizer=None, text_prefix_type="caption"):
    """
    Modify modality-indicating markers: remove, i2c, c2i, swap, swap_start.
    text_prefix_type: 'caption' or 'document' — which text span prefix to locate.
    tokenizer required for text span localization.
    """
    assert ty in ["remove", "i2c", "c2i", "swap", "swap_start"]
    assert tokenizer is not None, "tokenizer required for text span localization"
    tok = SPECIAL_TOKENS[model_family]

    out = {k: v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    device = input_ids.device
    dtype = input_ids.dtype
    B, L = input_ids.shape

    if model_family == "qwen2.5":
        img_start_id = int(tok["vision_start"])
        img_end_id = int(tok["vision_end"])
        image_token_id = int(tok["image_pad"])
    elif model_family == "internvl":
        img_start_id = int(tok["img_start"])
        img_end_id = int(tok["img_end"])
        image_token_id = int(tok["img_context"])
    elif model_family == "gemma":
        img_start_id = int(tok["boi"])
        img_end_id = int(tok["eoi"])
        image_token_id = int(tok["image_token"])
    else:
        raise ValueError(f"Unknown model_family={model_family}")

    text_prefix_options = get_text_prefix_from_tokenizer(tokenizer, text_prefix_type)
    text_prefix_options = [p.to(device=device, dtype=dtype).view(-1) for p in text_prefix_options]
    dot_id = int(tok["dot"])

    def _find_first(h, token_id, start=0):
        idxs = torch.where(h[start:] == token_id)[0]
        return None if idxs.numel() == 0 else int(idxs[0].item() + start)

    def _locate(seq):
        img_s = _find_first(seq, img_start_id, start=0)
        img_e = None
        if img_s is not None:
            img_e = _find_first(seq, img_end_id, start=img_s + 1)
            if img_e is None:
                raise ValueError("Found image_start but not image_end.")

        text_s = None
        actual_text_prefix_len = 0
        matched_prefix = None
        for text_prefix in text_prefix_options:
            text_s = _find_subseq(seq.detach().cpu(), text_prefix.detach().cpu(), start=0)
            if text_s is not None:
                actual_text_prefix_len = int(text_prefix.numel())
                matched_prefix = text_prefix
                break

        text_dot = None
        if text_s is not None:
            text_content_start = int(text_s) + actual_text_prefix_len
            text_dot = _first_token_after(seq.detach().cpu(), dot_id, start=text_content_start)
            if text_dot is None:
                raise ValueError('Found text prefix but not "." after it.')

        return img_s, img_e, (None if text_s is None else int(text_s)), (None if text_dot is None else int(text_dot)), actual_text_prefix_len, matched_prefix

    def _new_pos(old_pos, deleted_sorted):
        return old_pos - sum(1 for d in deleted_sorted if d < old_pos)

    def _delete_many(seq, delete_idx_set):
        keep = [i for i in range(seq.numel()) if i not in delete_idx_set]
        return seq[keep]

    def _insert(seq, pos, ins):
        return torch.cat([seq[:pos], ins, seq[pos:]], dim=0)

    need_tti = (model_family == "gemma") or ("token_type_ids" in inputs)
    new_tti_rows = [] if need_tti else None
    new_rows = []
    new_len = None

    for b in range(B):
        seq = input_ids[b]
        img_s, img_e, text_s, text_dot, actual_text_prefix_len, matched_prefix = _locate(seq)
        actual_text_prefix = matched_prefix if matched_prefix is not None else text_prefix_options[0]

        deletes = set()
        inserts = []

        if ty == "remove":
            if img_s is not None:
                deletes.add(img_s)
            if img_e is not None:
                deletes.add(img_e)
            if text_s is not None:
                for j in range(actual_text_prefix_len):
                    deletes.add(text_s + j)
            if text_dot is not None:
                deletes.add(text_dot)

        elif ty == "i2c":
            if img_s is None or img_e is None:
                raise ValueError("i2c requires an image block.")
            deletes.update([img_s, img_e])
            inserts.append((img_s, actual_text_prefix))
            inserts.append((img_e, torch.tensor([dot_id], device=device, dtype=dtype)))

        elif ty == "c2i":
            if text_s is None or text_dot is None:
                raise ValueError("c2i requires text prefix and '.' after it.")
            for j in range(actual_text_prefix_len):
                deletes.add(text_s + j)
            deletes.add(text_dot)
            inserts.append((text_s, torch.tensor([img_start_id], device=device, dtype=dtype)))
            inserts.append((text_dot, torch.tensor([img_end_id], device=device, dtype=dtype)))

        elif ty == "swap":
            if img_s is not None and img_e is not None:
                deletes.update([img_s, img_e])
                inserts.append((img_s, actual_text_prefix))
                inserts.append((img_e, torch.tensor([dot_id], device=device, dtype=dtype)))
            if text_s is not None and text_dot is not None:
                for j in range(actual_text_prefix_len):
                    deletes.add(text_s + j)
                deletes.add(text_dot)
                inserts.append((text_s, torch.tensor([img_start_id], device=device, dtype=dtype)))
                inserts.append((text_dot, torch.tensor([img_end_id], device=device, dtype=dtype)))
        elif ty == "swap_start":
            if img_s is not None and img_e is not None:
                deletes.update([img_s])
                inserts.append((img_s, actual_text_prefix))
            if text_s is not None and text_dot is not None:
                for j in range(actual_text_prefix_len):
                    deletes.add(text_s + j)
                inserts.append((text_s, torch.tensor([img_start_id], device=device, dtype=dtype)))

        deleted_sorted = sorted(deletes)
        base = _delete_many(seq, deletes)
        inserts_sorted = sorted(inserts, key=lambda x: x[0])
        shift = 0
        for old_pos, ins in inserts_sorted:
            pos = _new_pos(old_pos, deleted_sorted) + shift
            base = _insert(base, pos, ins)
            shift += int(ins.numel())

        if new_len is None:
            new_len = base.numel()
        else:
            assert base.numel() == new_len
        new_rows.append(base)

        if need_tti:
            tti = torch.zeros((new_len,), device=device, dtype=inputs["token_type_ids"].dtype)
            tti[base == image_token_id] = 1
            new_tti_rows.append(tti)

    out["input_ids"] = torch.stack(new_rows, dim=0)
    if "attention_mask" in inputs:
        out["attention_mask"] = torch.ones((B, new_len), device=device, dtype=inputs["attention_mask"].dtype)
    if need_tti:
        out["token_type_ids"] = torch.stack(new_tti_rows, dim=0)

    return out


def substitute_image_with_text_substrate(
    inputs,
    image_labels,
    tokenizer,
    model_family="qwen2.5",
    text_labels=None,
    text_prefix_type="caption",
):
    """
    Substitute image content tokens with textual substrate.
    text_prefix_type: 'caption' or 'document' — which text span to use.
    text_labels: caption or document labels for each batch item.
    """
    from utils.parse_spans_utils import _find_subseq, _first_token_after

    assert len(image_labels) == inputs["input_ids"].shape[0]
    if text_labels is not None:
        assert len(text_labels) == inputs["input_ids"].shape[0]
    tok = SPECIAL_TOKENS[model_family]

    out = {k: v for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    device = input_ids.device
    dtype = input_ids.dtype
    B, L = input_ids.shape

    if model_family == "qwen2.5":
        img_start_id = int(tok["vision_start"])
        img_end_id = int(tok["vision_end"])
        image_token_id = int(tok["image_pad"])
    elif model_family == "internvl":
        img_start_id = int(tok["img_start"])
        img_end_id = int(tok["img_end"])
        image_token_id = int(tok["img_context"])
    elif model_family == "gemma":
        img_start_id = int(tok["boi"])
        img_end_id = int(tok["eoi"])
        image_token_id = int(tok["image_token"])
    else:
        raise ValueError(f"Unknown model_family={model_family}")

    text_prefix_options = get_text_prefix_from_tokenizer(tokenizer, text_prefix_type)
    text_prefix_options = [p.to(device=device, dtype=dtype).view(-1) for p in text_prefix_options]
    dot_id = int(tok["dot"])

    def _find_first(h, token_id, start=0):
        idxs = torch.where(h[start:] == token_id)[0]
        return None if idxs.numel() == 0 else int(idxs[0].item() + start)

    def _locate(seq):
        img_s = _find_first(seq, img_start_id, start=0)
        img_e = None
        if img_s is not None:
            img_e = _find_first(seq, img_end_id, start=img_s + 1)
            if img_e is None:
                raise ValueError("Found image_start but not image_end.")

        text_s = None
        actual_text_prefix_len = 0
        for text_prefix in text_prefix_options:
            text_s = _find_subseq(seq.detach().cpu(), text_prefix.detach().cpu(), start=0)
            if text_s is not None:
                actual_text_prefix_len = int(text_prefix.numel())
                break

        text_dot = None
        if text_s is not None:
            text_content_start = int(text_s) + actual_text_prefix_len
            text_dot = _first_token_after(seq.detach().cpu(), dot_id, start=text_content_start)
            if text_dot is None:
                raise ValueError('Found text prefix but not "." after it.')

        return img_s, img_e, (None if text_s is None else int(text_s)), (None if text_dot is None else int(text_dot)), actual_text_prefix_len

    new_rows = []
    for b in range(B):
        seq = input_ids[b]
        img_s, img_e, text_s, text_dot, actual_text_prefix_len = _locate(seq)

        if img_s is None or img_e is None:
            new_seq = seq.clone()
        else:
            if (
                text_labels is not None
                and text_labels[b] is not None
                and text_s is not None
                and text_dot is not None
            ):
                text_content_start = text_s + actual_text_prefix_len
                text_content_ids = seq[text_content_start:text_dot].detach().cpu().tolist()
                text_content_str = tokenizer.decode(text_content_ids, skip_special_tokens=True)
                text_label_str = text_labels[b]
                if text_label_str in text_content_str:
                    image_content_text = text_content_str.replace(text_label_str, image_labels[b], 1)
                else:
                    image_content_text = " " + image_labels[b]
                content_tokens = tokenizer.encode(image_content_text, add_special_tokens=False)
            else:
                content_tokens = tokenizer.encode(" " + image_labels[b], add_special_tokens=False)
            content_tensor = torch.tensor(content_tokens, device=device, dtype=dtype)
            new_seq = torch.cat([seq[: img_s + 1], content_tensor, seq[img_e:]], dim=0)

        new_rows.append(new_seq)

    new_lens = [r.numel() for r in new_rows]
    max_len = max(new_lens)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer must have pad_token_id or eos_token_id")

    padded = []
    for r in new_rows:
        pad_len = max_len - r.numel()
        if pad_len > 0:
            r = torch.cat([torch.full((pad_len,), pad_id, device=r.device, dtype=r.dtype), r], dim=0)
        padded.append(r)
    out["input_ids"] = torch.stack(padded, dim=0)

    if "attention_mask" in inputs:
        attention = []
        for i in range(B):
            mask = torch.ones(new_lens[i], device=device, dtype=inputs["attention_mask"].dtype)
            pad_len = max_len - new_lens[i]
            if pad_len > 0:
                mask = torch.cat([torch.zeros(pad_len, device=device, dtype=mask.dtype), mask], dim=0)
            attention.append(mask)
        out["attention_mask"] = torch.stack(attention, dim=0)

    if (model_family == "gemma") or ("token_type_ids" in inputs):
        tti_dtype = inputs.get("token_type_ids")
        tti_dtype = tti_dtype.dtype if tti_dtype is not None else torch.long
        out["token_type_ids"] = torch.zeros((B, max_len), device=device, dtype=tti_dtype)

    return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}


def get_deleted_indices_for_remove(inputs, model_family="qwen2.5", tokenizer=None, text_prefix_type="caption"):
    """
    Return sorted list of token indices that would be deleted by modify_markers(..., ty="remove").
    Used to compute span positions after remove.
    """
    assert tokenizer is not None
    tok = SPECIAL_TOKENS[model_family]
    input_ids = inputs["input_ids"]
    seq = input_ids[0].detach().cpu() if input_ids.dim() > 1 else input_ids.detach().cpu()

    if model_family == "qwen2.5":
        img_start_id = int(tok["vision_start"])
        img_end_id = int(tok["vision_end"])
    elif model_family == "internvl":
        img_start_id = int(tok["img_start"])
        img_end_id = int(tok["img_end"])
    elif model_family == "gemma":
        img_start_id = int(tok["boi"])
        img_end_id = int(tok["eoi"])
    else:
        raise ValueError(f"Unknown model_family={model_family}")

    text_prefix_options = get_text_prefix_from_tokenizer(tokenizer, text_prefix_type)
    dot_id = int(tok["dot"])

    def _find_first(h, token_id, start=0):
        idxs = torch.where(h[start:] == token_id)[0]
        return None if idxs.numel() == 0 else int(idxs[0].item() + start)

    img_s = _find_first(seq, img_start_id, start=0)
    img_e = _find_first(seq, img_end_id, start=(img_s + 1) if img_s is not None else 0) if img_s is not None else None

    text_s = None
    actual_text_prefix_len = 0
    for text_prefix in text_prefix_options:
        text_s = _find_subseq(seq, text_prefix.detach().cpu() if hasattr(text_prefix, "detach") else text_prefix, start=0)
        if text_s is not None:
            actual_text_prefix_len = int(text_prefix.numel()) if hasattr(text_prefix, "numel") else len(text_prefix)
            break

    text_dot = None
    if text_s is not None:
        text_content_start = int(text_s) + actual_text_prefix_len
        text_dot = _first_token_after(seq, dot_id, start=text_content_start)

    deletes = set()
    if img_s is not None:
        deletes.add(img_s)
    if img_e is not None:
        deletes.add(img_e)
    if text_s is not None:
        for j in range(actual_text_prefix_len):
            deletes.add(text_s + j)
    if text_dot is not None:
        deletes.add(text_dot)

    return sorted(deletes)


def compute_span_positions_after_remove(
    image_span_positions,
    caption_span_positions,
    deleted_sorted,
):
    """Map span positions from clean input to positions after remove."""
    def _shift(pos_list):
        return [p - sum(1 for d in deleted_sorted if d < p) for p in pos_list]

    return _shift(image_span_positions), _shift(caption_span_positions)


def map_span_type(span_type):
    """Map span_type ('all','start','content','end') to (image_span_name, cap_span_name)."""
    return ("image_" + span_type, "cap_" + span_type)


def get_transformer_layers(model):
    """Get the list of transformer layers (tries common paths for different model architectures)."""
    for path in ["language_model.layers", "model.layers", "model.model.layers", "layers"]:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Could not find transformer layers in model")


def get_num_hidden_layers(model):
    """Get number of hidden layers from model config."""
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model, "text_config") and hasattr(model.text_config, "num_hidden_layers"):
        return model.text_config.num_hidden_layers
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        return model.config.text_config.num_hidden_layers
    raise AttributeError("Could not determine num_hidden_layers")

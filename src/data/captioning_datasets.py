"""Image captioning datasets: MSCOCO 2017 and Flickr30k."""

import os
import re
import random
from collections import defaultdict

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

from utils.prompt_evaluation_utils import get_paired_caption, get_prompt, _to_caption_strings

SEED = 42

# Paths for MSCOCO 2017. Point COCO_ROOT at a directory laid out as the
# official release, i.e. containing train2017/, val2017/ and annotations/.
COCO_ROOT = os.environ.get("COCO_ROOT", os.path.join(os.path.expanduser("~"), "data", "coco"))

MSCOCO_ROOT_VAL = os.path.join(COCO_ROOT, "val2017")
MSCOCO_ANN_VAL = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")
MSCOCO_ROOT_TRAIN = os.path.join(COCO_ROOT, "train2017")
MSCOCO_ANN_TRAIN = os.path.join(COCO_ROOT, "annotations", "captions_train2017.json")

# MSCOCO train (~118K total)
MSCOCO_TRAIN_SIZE = 4000
# Split sizes: MSCOCO val (5K total)
MSCOCO_VAL_SIZE = 2000
MSCOCO_TEST_SIZE = 2000

# Flickr30k
FLICKR30K_TRAIN_SIZE = 4000
FLICKR30K_VAL_SIZE = 2000
FLICKR30K_TEST_SIZE = 2000


def _format_caption(s: str) -> str:
    """Normalize caption: collapse spaces, ensure ends with period, strip quotes."""
    if not s or not isinstance(s, str):
        return s
    s = s.strip().strip('"\'')
    s = re.sub(r"\s+", " ", s)  # replace multiple spaces/newlines with single space
    if s and not s.endswith("."):
        s = s.rstrip(".,;:!?") + "."
    s = re.sub(r"\s+\.$", ".", s)  # remove space before trailing period
    return s.strip('"\'')


def _format_captions(captions):
    """Format a list of caption strings (or single string)."""
    if captions is None:
        return []
    if isinstance(captions, str):
        return [_format_caption(captions)]
    return [_format_caption(c) for c in captions if c]


def _rescale_image_content(image, scale_factor, keep_size=False):
    """Downscale an image by a given scale factor."""
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    if keep_size:
        image = image.resize((original_width, original_height), Image.Resampling.LANCZOS)
    return image


class MSCOCOCaptioningDataset(Dataset):
    """
    MSCOCO 2017 captions. Supports val split (3K val, 2K test) or train split (10K train, 10K val, 2K test).
    CocoCaptions returns (image, target) where target has 'annotations' with caption dicts.
    """

    def __init__(self, root=None, ann_file=None, split="test", coco_split="val", seed=SEED):
        if coco_split == "val":
            root = root or MSCOCO_ROOT_VAL
            ann_file = ann_file or MSCOCO_ANN_VAL
        else:
            root = root or MSCOCO_ROOT_TRAIN
            ann_file = ann_file or MSCOCO_ANN_TRAIN
        self.ds = CocoCaptions(root=root, annFile=ann_file)
        self.split = split
        self.coco_split = coco_split
        total = len(self.ds)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(total)
        if coco_split == "val":
            if split == "val":
                self.indices = indices[:MSCOCO_VAL_SIZE]
            elif split == "test":
                self.indices = indices[MSCOCO_VAL_SIZE : MSCOCO_VAL_SIZE + MSCOCO_TEST_SIZE]
            else:
                raise ValueError(f"MSCOCO val split only supports 'val' or 'test', got {split}")
        else:
            needed = MSCOCO_TRAIN_SIZE + FLICKR30K_VAL_SIZE + FLICKR30K_TEST_SIZE
            if total < needed:
                raise ValueError(f"MSCOCO train has {total} samples, need {needed}")
            if split == "train":
                self.indices = indices[:MSCOCO_TRAIN_SIZE]
            elif split == "val":
                self.indices = indices[MSCOCO_TRAIN_SIZE : MSCOCO_TRAIN_SIZE + FLICKR30K_VAL_SIZE]
            elif split == "test":
                self.indices = indices[
                    MSCOCO_TRAIN_SIZE
                    + FLICKR30K_VAL_SIZE : MSCOCO_TRAIN_SIZE
                    + FLICKR30K_VAL_SIZE
                    + FLICKR30K_TEST_SIZE
                ]
            else:
                raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.ds[int(self.indices[idx])]


class Flickr30kCaptioningDataset(Dataset):
    """
    Flickr30k from HuggingFace (lmms-lab/flickr30k).
    Random indices: 10K train, 10K val, 2K test (uniformly sampled).
    """

    def __init__(self, split="test", seed=SEED):
        from datasets import load_dataset

        ds = load_dataset("lmms-lab/flickr30k", split="test")
        total = len(ds)
        needed = FLICKR30K_TRAIN_SIZE + FLICKR30K_VAL_SIZE + FLICKR30K_TEST_SIZE
        if total < needed:
            raise ValueError(f"Flickr30k train has {total} samples, need {needed}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(total)
        if split == "train":
            split_indices = indices[:FLICKR30K_TRAIN_SIZE]
        elif split == "val":
            split_indices = indices[
                FLICKR30K_TRAIN_SIZE : FLICKR30K_TRAIN_SIZE + FLICKR30K_VAL_SIZE
            ]
        elif split == "test":
            split_indices = indices[
                FLICKR30K_TRAIN_SIZE
                + FLICKR30K_VAL_SIZE : FLICKR30K_TRAIN_SIZE
                + FLICKR30K_VAL_SIZE
                + FLICKR30K_TEST_SIZE
            ]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.samples = []
        for i in tqdm(split_indices, desc=f"Loading Flickr30k {split}"):
            self.samples.append((ds["image"][int(i)], ds["caption"][int(i)]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MultimodalCaptioningDataset(Dataset):
    """
    Wrapper for captioning datasets. Each sample: image + 5 captions.
    For consistent: paired_caption = random from same image's 5.
    For inconsistent: paired_caption = random from another image's 5.
    """

    def __init__(
        self,
        args,
        split,
        prompt_generator,
        image_content_rescaling_factor=1.0,
        keep_original_size=False,
        encoder=None,
    ):
        self.args = args
        self.split = split
        self.prompt_generator = prompt_generator
        self.encoder = encoder
        self.image_content_rescaling_factor = image_content_rescaling_factor
        self.keep_original_size = keep_original_size

        self.data = self._get_data()
        encoder = getattr(self, "encoder", None)
        self.all_sources, self.all_image_captions, self.all_paired_captions = self._process_data(
            encoder=encoder
        )

    def __len__(self):
        return len(self.data)

    def _get_data(self):
        seed = getattr(self.args, "seed", SEED)
        if self.args.dataset == "mscoco":
            coco_split = getattr(self.args, "mscoco_split", "val")
            if coco_split == "val":
                root = getattr(self.args, "mscoco_root", None) or MSCOCO_ROOT_VAL
                ann = getattr(self.args, "mscoco_ann", None) or MSCOCO_ANN_VAL
            else:
                root = getattr(self.args, "mscoco_root_train", None) or MSCOCO_ROOT_TRAIN
                ann = getattr(self.args, "mscoco_ann_train", None) or MSCOCO_ANN_TRAIN
            return MSCOCOCaptioningDataset(
                root=root,
                ann_file=ann,
                split=self.split,
                coco_split=coco_split,
                seed=seed,
            )
        elif self.args.dataset == "flickr30k":
            return Flickr30kCaptioningDataset(split=self.split, seed=seed)
        raise NotImplementedError(f"{self.args.dataset} not supported. Use: mscoco, flickr30k.")

    def _process_data(self, encoder=None):
        all_sources, all_image_captions, all_paired_captions = [], [], []
        all_captions_by_image = [self.data[i][1] for i in range(len(self.data))]
        random.seed(SEED)
        np.random.seed(SEED)
        for ex_id in tqdm(range(len(self.data)), desc=f"{self.args.dataset} {self.split}"):
            image, captions = self.data[ex_id]
            paired_caption = get_paired_caption(
                self.args, captions, ex_id, all_captions_by_image, encoder=encoder
            )
            img_caption_strs = _format_captions(_to_caption_strings(captions))
            paired_caption = _format_caption(paired_caption) if paired_caption else None
            prompt = get_prompt(self.args, self.prompt_generator, paired_caption)
            all_sources.append(prompt)
            all_image_captions.append(img_caption_strs)
            all_paired_captions.append(paired_caption)
        return all_sources, all_image_captions, all_paired_captions

    def get_image(self, idx):
        image = self.data[idx][0]
        if self.image_content_rescaling_factor != 1.0:
            image = _rescale_image_content(
                image,
                self.image_content_rescaling_factor,
                self.keep_original_size,
            )
        return image

    def __getitem__(self, idx):
        return {
            "source": self.all_sources[idx],
            "image_captions": self.all_image_captions[idx],
            "paired_caption": self.all_paired_captions[idx],
            "image": self.get_image(idx) if self.args.model_family != "gemma" else [self.get_image(idx)],
        }


class LmEvaluationDataCollator:
    def __init__(self, processor, text_only=False):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.is_text_only = text_only

    def __call__(self, features, return_tensors="pt"):
        raw_texts = [f["source"] for f in features]
        images = [f["image"] for f in features]

        if self.is_text_only:
            inputs = self.processor.tokenizer(raw_texts, padding=True, return_tensors="pt")
        else:
            inputs = self.processor(
                text=raw_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        image_captions = [f["image_captions"] for f in features]
        paired_captions = [f["paired_caption"] for f in features]

        return {
            "inputs": {**inputs},
            "image_captions": image_captions,
            "paired_captions": paired_captions,
        }

from torch.utils.data import DataLoader

from data.captioning_datasets import MultimodalCaptioningDataset, LmEvaluationDataCollator

CAPTIONING_DATASETS = ["mscoco", "flickr30k"]


def get_dataset(split, prompt_generator, args, encoder=None):
    if args.dataset in CAPTIONING_DATASETS:
        data_wrapper = MultimodalCaptioningDataset
    else:
        raise NotImplementedError(
            f"{args.dataset} not supported. Use: {CAPTIONING_DATASETS}."
        )

    dataset = data_wrapper(args, split, prompt_generator, encoder=encoder)
    print(f"dataset_{split}: {len(dataset)}")
    return dataset


def get_dataloader(dataset, processor, args, is_train):
    if args.dataset in CAPTIONING_DATASETS:
        collator = LmEvaluationDataCollator(
            processor,
            text_only=(getattr(args, "input_type", "inconsistent") == "text_only"),
        )
    else:
        raise NotImplementedError(
            f"{args.dataset} not supported. Use: {CAPTIONING_DATASETS}."
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        pin_memory=False,
        shuffle=True if is_train else False,
        drop_last=True if is_train else False,
        num_workers=0,
    )

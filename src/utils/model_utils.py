def load_model_and_preprocess(args):
    import torch

    if args.model_family == "llava":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    elif args.model_family == "instructblip":
        from transformers import InstructBlipProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = InstructBlipProcessor.from_pretrained(args.model_name, use_fast=True)

        if args.input_type == "text_only":
            model = model.language_model

    elif args.model_family == "llava-onevision":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)
        processor.tokenizer.padding_side = "left"

    elif args.model_family == "qwen2.5":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        min_pixels = 32 * 32
        max_pixels = 512 * 512
        processor = AutoProcessor.from_pretrained(args.model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
        processor.tokenizer.padding_side = "left"

    elif args.model_family in ["gemma", "internvl"]:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=f"{args.work_dir}/.cache/huggingface/hub",
        )
        processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)
        processor.tokenizer.padding_side = "left"

    else:
        raise NotImplementedError(f"Model family {args.model_family} not supported.")

    return model, processor


def load_bert_judge(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=None):
    """Load BERT-based sentence encoder for embedding-based 3-way classification."""
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_folder"] = cache_dir
    model = SentenceTransformer(model_name, **kwargs)
    return model

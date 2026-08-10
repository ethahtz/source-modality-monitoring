class AnyClass:
    def __init__(self):
        pass


def dict_to_object(d):
    obj = AnyClass()
    for k, v in d.items():
        setattr(obj, k, v)
    return obj


def get_model_family(args):
    """Supported: qwen2.5, llava, llava-onevision, instructblip, gemma (gemma3), internvl (InternVL3)."""
    if args.model_name.startswith("llava-hf/llava-1.5-7b-hf"):
        return "llava"
    elif args.model_name.startswith("Salesforce/instructblip"):
        return "instructblip"
    elif args.model_name.startswith("Qwen/Qwen2.5-VL"):
        return "qwen2.5"
    elif args.model_name.startswith("google/gemma"):
        return "gemma"
    elif args.model_name.startswith("OpenGVLab/InternVL3"):
        return "internvl"
    elif args.model_name.startswith("llava-hf/llava-onevision"):
        return "llava-onevision"
    else:
        raise Exception(f"`{args.model_name}` is not supported. Use: qwen2.5, llava, llava-onevision, instructblip, gemma, internvl.")


def get_prompt_template_args(args):
    """Set prompt templates for open-ended evaluation (no multiple-choice)."""
    # Ensure required attributes exist (PromptGenerator needs these)
    if not hasattr(args, "caption_template"):
        args.caption_template = [" {} "]
    if not hasattr(args, "document_template"):
        args.document_template = [" {} "]
    if not hasattr(args, "text_template"):
        args.text_template = [" {} "]
    if not hasattr(args, "question_template"):
        args.question_template = ["What is in the image?", "Describe the image."]

    # Captioning datasets: mscoco, flickr30k (content used for both caption and document)
    captioning_datasets = ["mscoco", "flickr30k"]
    dataset = getattr(args, "dataset", None)

    if dataset in captioning_datasets:
        args.answer_template = "Answer:"
        args.further_instruction = "Answer the question with a short sentence."
    else:
        args.answer_template = getattr(args, "answer_template", "Answer:")
        args.further_instruction = getattr(args, "further_instruction", "Answer the question with a short sentence.")

    if dataset in captioning_datasets:
        args.caption_template = [" {} "]
        args.document_template = [" {} "]

        if args.modality_to_report == "image":
            args.question_template = [
                "What is in the image?",
                "Describe the image.",
            ]
        elif args.modality_to_report == "text":
            if getattr(args, "prompt_format", "image_caption") == "image_document":
                args.question_template = [
                    "What is in the document?",
                    "What does the document say?",
                ]
            elif getattr(args, "prompt_format", "image_caption") == "image_caption":
                args.question_template = [
                    "What does the caption say?",
                    "What is in the caption?",
                ]
            elif getattr(args, "prompt_format", "image_caption") == "image_text":
                args.question_template = [
                    "What is in the text?",
                    "What does the text say?",
                ]

    return args

"""Prompt generation for open-ended evaluation with image-caption and image-document inputs."""

from random import choice as rand_choice

IMAGE_POINTER = ""


class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.prompt_template = self._get_prompt_template()

    def _get_image_part(self):
        if getattr(self.args, "input_type", "inconsistent") == "text_only":
            return ""
        if self.args.model_family in ["llava", "llava-onevision"]:
            image_part = "<image>\n"
        elif self.args.model_family == "internvl":
            image_part = "<IMG_CONTEXT>\n"
        elif self.args.model_family == "gemma":
            image_part = "<start_of_image>\n"
        elif self.args.model_family == "qwen2.5":
            image_part = "<|vision_start|><|image_pad|><|vision_end|>"
        elif self.args.model_family == "instructblip":
            image_part = ""
        else:
            raise NotImplementedError(f"{self.args.model_family} is not supported yet.")
        if getattr(self.args, "use_pointers", 1) == 1:
            image_part = f"{IMAGE_POINTER}{image_part}"
        return image_part

    def _get_text_part(self):
        """Caption or Document part depending on prompt_format. Empty for image_only."""
        if getattr(self.args, "input_type", "inconsistent") == "image_only":
            return ""
        if self.args.prompt_format == "image_document":
            prefix = "Document:" if getattr(self.args, "use_pointers", 1) == 1 else ""
            template = rand_choice(self.args.document_template)
        elif self.args.prompt_format == "image_text":
            prefix = "Text:" if getattr(self.args, "use_pointers", 1) == 1 else ""
            template = rand_choice(self.args.text_template)
        elif self.args.prompt_format == "image_caption":
            prefix = "Caption:" if getattr(self.args, "use_pointers", 1) == 1 else ""
            template = rand_choice(self.args.caption_template)
        else:
            raise ValueError(f"Prompt format {self.args.prompt_format} not supported.")

        return f"{prefix}{template}"

    def _get_question_part(self):
        question_part = rand_choice(self.args.question_template)
        question_part += " " + self.args.further_instruction + " "
        question_part = f"Question: {question_part}"
        return question_part

    def _get_answer_part(self):
        return self.args.answer_template

    def _get_parts(self):
        return {
            "image": self._get_image_part(),
            "text": self._get_text_part(),
            "question": self._get_question_part(),
            "answer": self._get_answer_part(),
        }

    def _get_prompt_template(self):
        parts = self._get_parts()
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
            elif self.args.model_family == "llava-onevision":
                return f"<|im_start|>user {image_text_question}\n<|im_end|><|im_start|>assistant\n{parts['answer']}"
            elif self.args.model_family in ["llava", "instructblip"]:
                return f"USER: {image_text_question}\nASSISTANT: {parts['answer']}"
            else:
                raise RuntimeError(f"Model family {self.args.model_family} not supported.")
        else:
            return image_text_question + parts["answer"]

    def __call__(self, format_args=None):
        prompt_template = self._get_prompt_template()
        if format_args is not None:
            return prompt_template.format(*format_args)
        return prompt_template

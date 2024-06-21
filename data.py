import torch
from datasets import load_dataset
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<DocVQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image


class TheCauldronDataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split=split)
        self.examples = self._create_examples()
        self.task_prompt = "<Cauldron>"

    def _create_examples(self):
        examples = []
        for idx, item in enumerate(self.data):
            for qa_pair in item["texts"]:
                question_data = qa_pair.get("user", "")
                answer_data = qa_pair.get("assistant", "")
                examples.append((question_data, answer_data, idx))
        return examples

    def __getitem__(self, idx):
        (question_data, answer_data, image_idx) = self.examples[idx]

        question = self.task_prompt + self.correct_casing_finqa(question_data, True)
        answer = self.correct_casing_finqa(answer_data)

        image = self.data[image_idx]["images"][0]  # Fetch the image by index

        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image

import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

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
    
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + self.correct_casing_finqa(example['question'], True)
        first_answer = example['answers'][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example['image']  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image

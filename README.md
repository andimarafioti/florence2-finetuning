# Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models
This codebase supports a [blog we published on Huggingface.co on June 24th!](https://huggingface.co/blog/finetune-florence2)

Florence 2, released by Microsoft in June 2024, is a foundation vision-language model. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks.

Florence supports captioning, object detection, OCR, and more out of the box. However, your task might not be supported, or you might need to control the model's output for your task. That's when you will need to fine-tune the model.

In this repository, we present code to fine tune Florence on [DocVQA](https://www.docvqa.org/) or [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron).


## Installation

We use UV to manage packages. UV is an extremely fast Python package installer and resolver, written in Rust. You can get [it here](https://github.com/astral-sh/uv/).

To get started, run the following commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you encounter issues with flash-attn, you can fix it with the following command:

```bash
uv pip install -U flash-attn --no-build-isolation
```

## Get the data

For this experiment, we use the DocVQA dataset. Our team at Hugging Face has already uploaded a version to the hub, so you can use it directly.

The dataset is preprocessed for easy handling. After loading it with HF's dataset library, it looks like this:

```python
from datasets import load_dataset

data = load_dataset('HuggingFaceM4/DocumentVQA')

print(data)
```

Output:

```python
DatasetDict({
    train: Dataset({
        features: ['questionId', 'question', 'question_types', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'answers'],
        num_rows: 39463
    })
    validation: Dataset({
        features: ['questionId', 'question', 'question_types', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'answers'],
        num_rows: 5349
    })
    test: Dataset({
        features: ['questionId', 'question', 'question_types', 'image', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'answers'],
        num_rows: 5188
    })
})
```

## Updating Florence-2 for Fine-Tuning

To fine-tune Florence-2, we had to make some modifications to the `Florence2Seq2SeqLMOutput` class. We have submitted pull requests (PRs) with these changes, and you can find links to them in the code. To use the revised version, you need to specify the appropriate revision when loading the model:

```python
model = AutoModelForCausalLM.from_pretrained(
        "andito/Florence-2-large-ft", trust_remote_code=True
    ).to(device)
alternative_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", trust_remote_code=True, revision="refs/pr/10"
    ).to(device)
```

We have submitted PRs for the necessary fixes across all models from Microsoft. If you prefer to use a different model, please refer to the corresponding revision we created.

## Single GPU training

To train with just one GPU, you can simply run:

```bash
python train.py
```

It will automatically train on the DocVQA dataset. Training on the cauldron using just one GPU is not recommended.

## Distributed training

The `distributed_train.py` script allows you to train the Florence-2 model using distributed data parallelism, which can significantly speed up the training process when using multiple GPUs. Below are the steps to use this script:

```bash
python distributed_train.py --dataset <dataset_name> --epochs <num_epochs> --eval-steps <evaluation_steps>
```

Example:

```bash
python distributed_train.py --dataset docvqa --epochs 10 --eval-steps 1000
```

- dataset_name: Name of the dataset to use (docvqa or cauldron).
- num_epochs: Number of epochs to train (default is 10).
- evaluation_steps: Frequency of evaluation during training (default is 10000 steps).

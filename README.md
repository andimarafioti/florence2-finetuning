# Florence2-Finetuning
A quick exploration into fine-tuning Florence-2 on the DocVQA dataset.

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

## Fixing Florence-2 code

To fine-tune Florence-2, we needed to modify the `Florence2Seq2SeqLMOutput` class. Our PRs are open and the code already links to them, in general you need to pass the revision to the model call:

```python
model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", trust_remote_code=True, revision="refs/pr/10"
    ).to(device)
```

# florence2-finetuning
Quick exploration into fine tuning florence 2

## Installation

I'm using UV to manage packages. UV is an extremely fast Python package installer and resolver, written in Rust. Get [it here](https://github.com/astral-sh/uv/).

To get started, do:

uv venv
source .venv/bin/activate
uv install requirements.txt

If you have issues with flass-attn, this fixed it for me:

uv pip install -U flash-attn --no-build-isolation

## Get the data

For this experiment, I will use DocVQA. Our team at Hugging Face already uploaded a version to the hub, so you can use it directly.

The dataset is already preprocessed for easy processing. After loading it with HF's dataset library, it looks like this:

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


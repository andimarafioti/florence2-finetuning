import argparse
import os
from functools import partial

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

import wandb
from data import DocVQADataset, TheCauldronDataset, VQAInstructDataset
from peft import LoraConfig, get_peft_model


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=800
    ).to(device)
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size//2,
            collate_fn=partial(collate_fn, processor=processor, device=device),
            num_workers=num_workers,
            sampler=val_sampler,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders

def evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count):
    if rank == 0:
        avg_train_loss = train_loss / (global_step*batch_size*world_size)
        wandb.log({"step": global_step, "train_loss": avg_train_loss})
        print(f"Rank {rank} - Average Training Loss: {avg_train_loss}")

    # Evaluation phase
    model.eval()
    for val_name, val_loader in val_loaders.items():
        val_loss = 0
        with torch.no_grad():
            val_item_count = 0
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                val_item_count += len(batch)
                inputs, answers = batch

                # Prepare the input and target tensors
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                    truncation=True,
                    max_length=800,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()
                if val_item_count > max_val_item_count:
                    break

        avg_val_loss = val_loss / val_item_count
        print(f"Rank {rank} - Step {global_step} - Average Validation Loss ({val_name}): {avg_val_loss}")

        # Log metrics to wandb
        if rank == 0:
            wandb.log({f"{val_name}_val_loss": avg_val_loss, "step": global_step})

    model.train()

def train_model(rank, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Initialize wandb
    if rank == 0:  # Only initialize wandb in the main process
        wandb.init(project="DocVQA-instruct", name=run_name)
        wandb.config.update({
            "dataset": dataset_name,
            "batch_size": batch_size,
            "use_lora": use_lora,
            "epochs": epochs,
            "learning_rate": lr,
            "eval_steps": eval_steps,
            "world_size": world_size,
        })

    # Load the dataset based on the dataset_name argument
    if dataset_name == "docvqa":
        train_dataset = DocVQADataset(split='train')
        val_datasets = {"docvqa": DocVQADataset(split='validation')}
    elif dataset_name == "cauldron":
        train_dataset = TheCauldronDataset(split='train')
        val_datasets = {
            "cauldron": TheCauldronDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'vqainstruct':
        train_dataset = VQAInstructDataset(split='train')
        val_datasets = {
            "vqainstruct": VQAInstructDataset(split='validation'), 
            "docvqa": DocVQADataset(split='validation')
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        "andito/Florence-2-large-ft", trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "andito/Florence-2-large-ft", trust_remote_code=True
    )

    if use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[rank])

    # Create DataLoaders
    num_workers = 0
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count)

        evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count)

        # Log training loss to wandb
        avg_train_loss = train_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})

        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            output_dir = f"./model_checkpoints/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    # Finish the wandb run
    if rank == 0:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["docvqa", "cauldron", "vqainstruct"], help="Dataset to train on")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=1000, help="Maximum number of items to evaluate on during validation")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args.dataset, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()

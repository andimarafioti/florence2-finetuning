from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, AutoModelForCausalLM, AdamW, get_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import os
from functools import partial
from data import DocVQADataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, rank, world_size, processor, device):
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, processor=processor, device=device), num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, processor=processor, device=device), num_workers=num_workers, sampler=val_sampler)
    
    return train_loader, val_loader

def train_model(rank, world_size, epochs=3, lr=5e-5):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load the dataset
    data = load_dataset("HuggingFaceM4/DocumentVQA")
    
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = DDP(model, device_ids=[rank])
    
    # Create datasets
    train_dataset = DocVQADataset(data['train'])
    val_dataset = DocVQADataset(data['validation'])

    # Create DataLoaders
    batch_size = 16
    num_workers = 4
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, rank, world_size, processor, device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank):
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Rank {rank} - Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}", position=rank):
                inputs, answers = batch

                # Prepare the input and target tensors
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Rank {rank} - Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        if rank == 0:  # Only the main process saves the checkpoint
            output_dir = f"./model_checkpoints/version{2}/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()

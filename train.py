import random
import warnings
import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from config import get_config, get_weights_file_path, latest_weights_file_path
from model import build_and_wrap_transformer
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from torchmetrics.text import CharErrorRate, WordErrorRate

from torch.utils.data import DataLoader, random_split
from data import OpusBooks, causal_mask

import wandb

from pathlib import Path

from datasets import load_dataset

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "<PRIVATE_IP_ADDRESS>"  # Replace with your master node's IP address
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)  # Set the device for each rank
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()
        
def save_checkpoint(model, optimizer, epoch, global_step, config, rank):
    if rank == 0:  # Only rank 0 saves the model checkpoint
        model_filename = get_weights_file_path(config, f'epoch:{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

def get_sentences(dataset, lang):
    for sample in dataset:
        yield sample['translation'][lang]

def get_dataset(config, rank, world_size):
    dataset = load_dataset("Helsinki-NLP/opus_books", f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    # Build tokenizer
    src_tokenizer = build_tokenizer(config, dataset, config['src_lang'])
    tgt_tokenizer = build_tokenizer(config, dataset, config['tgt_lang'])

    if rank == 0:  # Only print info on main process
        print(f"Source tokenizer vocab size: {src_tokenizer.get_vocab_size()}")
        print(f"Target tokenizer vocab size: {tgt_tokenizer.get_vocab_size()}")

    num_train_samples = int(0.9 * len(dataset))
    num_valid_samples = len(dataset) - num_train_samples

    # Use generator to ensure all ranks get the same split
    generator = torch.Generator().manual_seed(42)
    train_df, val_df = random_split(dataset=dataset, 
                                    lengths=[num_train_samples, num_valid_samples],
                                    generator=generator)

    train_dataset = OpusBooks(train_df, config['src_lang'], config['tgt_lang'], src_tokenizer, tgt_tokenizer, config['seq_len'])
    val_dataset = OpusBooks(val_df, config['src_lang'], config['tgt_lang'], src_tokenizer, tgt_tokenizer, config['seq_len'])

    max_src_len, max_tgt_len = 0, 0

    for item in dataset:
        src_ids = src_tokenizer.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['tgt_lang']]).ids
        max_src_len = max(max_src_len, len(src_ids))
        max_tgt_len = max(max_tgt_len, len(tgt_ids))
    
    print(f"Max length of source and target sentences : {max_src_len} , {max_tgt_len}")

    # Partition the dataset for distributed training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Optimized DataLoader settings
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batchsize'], 
        sampler=train_sampler
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, 
        sampler=val_sampler
    )

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer'].format(lang))

    # Load if file exists
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer and "[SOS]" in tokenizer.get_vocab():
            return tokenizer  # Return if valid

    # Retrain if file is missing or invalid
    tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
    tokenizer.train_from_iterator(get_sentences(dataset, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    
    return tokenizer

def greedy_decode(model, source, source_mask, tgt_tokenizer, max_len, device):
    # Get the base model if we're dealing with DDP model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module  # Unwrap DDP model
    
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.linearlayer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

import torch
import torch.distributed as dist
import random
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_model(model, validation_dataloader, tokenizer_tgt, max_len, device, global_step, epoch, rank=0, world_size=1):
    model.eval()

    source_texts, targets, preds = [], [], []
    total_bleu, total_rougeL, num_examples = 0, 0, 0
    smooth = SmoothingFunction().method1

    with torch.no_grad():
        progress_bar = tqdm(validation_dataloader, desc=f'Validation Epoch {epoch}', disable=(rank != 0))

        for batch in progress_bar:  
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Ensure batch size is always 1 per GPU
            if encoder_input.size(0) != 1:
                continue

            # Generate output
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_txt"][0]
            target_text = batch["tgt_txt"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            targets.append(target_text)
            preds.append(model_out_text)

            # Compute BLEU and ROUGE-L scores
            bleu_score = sentence_bleu([target_text.split()], model_out_text.split(), smoothing_function=smooth)
            rouge_score_val = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(target_text, model_out_text)

            # Accumulate scores
            total_bleu += bleu_score
            total_rougeL += rouge_score_val["rougeL"].fmeasure
            num_examples += 1

        # Prevent division by zero if no samples exist
        if num_examples > 0:
            avg_bleu = total_bleu / num_examples
            avg_rougeL = total_rougeL / num_examples
        else:
            avg_bleu, avg_rougeL = 0, 0

    # Synchronize across distributed processes
    if world_size > 1:
        gathered_source, gathered_targets, gathered_preds = [None] * world_size, [None] * world_size, [None] * world_size
        dist.all_gather_object(gathered_source, source_texts)
        dist.all_gather_object(gathered_targets, targets)
        dist.all_gather_object(gathered_preds, preds)

        if rank == 0:
            all_sources = [item for sublist in gathered_source for item in sublist]
            all_targets = [item for sublist in gathered_targets for item in sublist]
            all_preds = [item for sublist in gathered_preds for item in sublist]
    else:
        all_sources, all_targets, all_preds = source_texts, targets, preds

    # Logging to Weights & Biases
    if rank == 0 and wandb.run is not None:
        wandb.log({
            'valid/BLEU': avg_bleu,
            'valid/ROUGE-L': avg_rougeL,
            'global_step': global_step
        })

    # Print and save evaluation results
    if rank == 0 and all_sources:
        random_idx = random.randint(0, len(all_sources) - 1)

        with open('output.txt', 'a') as file:
            file.write(f"\n{'='*40} EPOCH {epoch} {'='*40}\n")
            file.write(f"Epoch: {epoch}\n")
            file.write(f"{'Source:':>12} {all_sources[random_idx]}\n")
            file.write(f"{'Target:':>12} {all_targets[random_idx]}\n")
            file.write(f"{'Pred:':>12} {all_preds[random_idx]}\n")
            file.write(f"BLEU Score: {avg_bleu:.4f}\n")
            file.write(f"ROUGE-L Score: {avg_rougeL:.4f}\n")
            file.write(f"{'-'*100}\n")


def train_model(rank, config, world_size):
    # Initialize distributed process group
    setup(rank, world_size)
    
    # Add a synchronization barrier
    dist.barrier()

    device = torch.device(f'cuda:{rank}')

    # Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config, rank, world_size)

    model = build_and_wrap_transformer(src_vocab=src_tokenizer.get_vocab_size(),
                                       tgt_vocab=tgt_tokenizer.get_vocab_size(),
                                       src_seq_len=config['seq_len'],
                                       tgt_seq_len=config['seq_len'],
                                       d_model=config['d_model'],
                                       rank=rank, world_size=world_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_function = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        state = torch.load(model_filename, map_location=device)
        model.module.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        if rank == 0:
            print(f'Loaded model state from {model_filename}')
    else:
        if rank == 0:
            print('No model to preload, starting from scratch')
            wandb.init(project="pytorch-transformer", config=config)   
            # define our custom x axis metric
            wandb.define_metric("global_step")
            # define which metrics will be plotted against it
            wandb.define_metric("valid/*", step_metric="global_step")
            wandb.define_metric("train/*", step_metric="global_step")

    # Add another synchronization point to ensure all processes have loaded model
    dist.barrier()

    for epoch in range(initial_epoch, config['epochs']):
        torch.cuda.empty_cache()
        model.train()
        
        # Set epoch for the sampler to reshuffle data
        train_dataloader.sampler.set_epoch(epoch)
        
        batchIterator = tqdm(train_dataloader, desc=f'Training epoch {epoch:02d}') if rank == 0 else train_dataloader
        
        for batch in batchIterator:
            encoder_ip = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_ip = batch['decoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (batch, 1, seq_len, seq_len)
            target_op = batch['targets'].to(device)  # (batch, seq_len)

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass through the model properly using its full interface
            # Important: Let the DDP wrapper handle the forward method
            encoder_output = model.module.encode(encoder_ip, encoder_mask)
            decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_ip, decoder_mask)
            proj_output = model.module.linearlayer(decoder_output)
            
            # Calculate loss
            loss = loss_function(proj_output.reshape(-1, tgt_tokenizer.get_vocab_size()), target_op.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            
            # Optimizer step
            optimizer.step()

            if rank == 0:
                batchIterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            
            global_step += 1
        

        evaluate_model(model, val_dataloader, tgt_tokenizer, config['seq_len'], device, global_step, epoch, rank)

        scheduler.step()

        save_checkpoint(model, optimizer, epoch, global_step, config, rank)

    cleanup()
        
def main():
    warnings.filterwarnings('ignore')
    config = get_config()
    config['preload'] = None
    world_size = torch.cuda.device_count()
    mp.spawn(train_model, args=(config, world_size), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

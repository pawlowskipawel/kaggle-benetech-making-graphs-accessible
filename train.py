from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

from benetech.training import BenetechTrainer, get_optimizer, freeze_encoder
from benetech.data import BenetechDataset, DataCollatorCTCWithPadding
from benetech.models import DeplotWithClassificationHead
from benetech.utils import freeze_encoder
from benetech.conf import parse_cfg

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import random
import torch
import wandb
import os
import gc


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
          
if __name__ == "__main__":
    seed_everything()
    cfg = parse_cfg()
        
    if not cfg.cls_head:
        cfg.TRAIN_ANNOTATION_PATH = cfg.TRAIN_ANNOTATION_PATH.replace(".csv", "_with_type.csv")
    if cfg.extra_data:
        cfg.TRAIN_ANNOTATION_PATH = cfg.TRAIN_ANNOTATION_PATH.replace(".csv", "_extra.csv")
        
    df = pd.read_csv(cfg.TRAIN_ANNOTATION_PATH)
    processor = Pix2StructProcessor.from_pretrained(cfg.backbone, is_vqa=False)
    
    df["tokenized_text"] = df["target_text"].apply(lambda x: processor.tokenizer.encode(x, padding="longest", return_tensors="pt", add_special_tokens=True)[0])
    df = df[(df["tokenized_text"].apply(len) < cfg.max_token_len)].reset_index(drop=True)
    
    best_results_dict = {}
    
    train_df = df[df["source"] != "extracted"]
    valid_df = df[df["source"] == "extracted"]
    
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    train_dataset = BenetechDataset(train_df, processor, mode="train_val", max_patches=cfg.max_patches, transforms=cfg.train_transforms)
    valid_dataset = BenetechDataset(valid_df, processor, mode="train_val", max_patches=cfg.max_patches)
    
    data_collator = DataCollatorCTCWithPadding(processor)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, drop_last=True, collate_fn=data_collator)
    
    if cfg.cls_head:
        model = DeplotWithClassificationHead(cfg.backbone) 
    else:
        model = Pix2StructForConditionalGeneration.from_pretrained(cfg.backbone)
        model.config.text_config.is_decoder=True
    
    if cfg.freeze_encoder:
        freeze_encoder(model)
    
    model.to(cfg.device)
    
    optimizer = get_optimizer("adamw", model, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_dataloader_len = len(train_dataloader)
    steps_per_epoch = (train_dataloader_len // cfg.grad_accum_steps) + (1 if train_dataloader_len % cfg.grad_accum_steps else 0)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.max_learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.scheduler_warmup_epochs / cfg.epochs,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor)
        
    trainer = BenetechTrainer(model, processor, optimizer, cfg, lr_scheduler=lr_scheduler)
    
    best_results = trainer.train(cfg.epochs, train_dataloader, valid_dataloader)
    
    for key, value in best_results.items():
        best_results_dict[key] = best_results_dict.get(key, []) + [value]
            
    del model, optimizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

if cfg.wandb_log:
    for i, (key, values) in enumerate(best_results_dict.items()):
        wandb.run.summary[key] = np.array(values).mean()

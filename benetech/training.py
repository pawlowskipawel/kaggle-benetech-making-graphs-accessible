# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/training.ipynb.

# %% auto 0
__all__ = ['freeze_encoder', 'get_optimizer', 'BenetechTrainer']

# %% ../nbs/training.ipynb 1
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import os
import gc
import re

import numpy as np

from .utils import process_output
from .data import TYPE2INT

# %% ../nbs/training.ipynb 2
def freeze_encoder(model):
    """Freeze ConvNet encoder of the Whisper model

    Args:
        model (WhisperForConditionalGeneration): Whisper model
    """
    for n, p in model.named_parameters():
        if "encoder" in n:
            p.requires_grad = False

# %% ../nbs/training.ipynb 3
def get_optimizer(optimizer_name, model, learning_rate, weight_decay):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n in decay_parameters],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_parameters],
         "weight_decay": 0.0}]

    if optimizer_name == "adam":
        return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
    if optimizer_name == "adamw":
        return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    raise ValueError(f"Unknown optimizer name: {optimizer_name}")

# %% ../nbs/training.ipynb 4
class BenetechTrainer:
    def __init__(self, model, processor, optimizer, cfg, lr_scheduler=None):
        
        self.cfg = cfg
            
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

        self.grad_clip_norm = cfg.grad_clip_norm
        self.grad_accum_steps = cfg.grad_accum_steps
        
        self.cls_head = cfg.cls_head
        self.max_token_len = cfg.max_token_len
        
        self.fp16 = cfg.fp16
        self.device = cfg.device
        self.wandb_log = cfg.wandb_log
        self.save_path = os.path.join(cfg.save_path, cfg.config_name)
        
        self.validation_step = cfg.validation_step
        self.first_eval_epoch = cfg.first_eval_epoch
        
        self.metrics_dict = cfg.metrics_dict
        self.best_metrics = {metric: 0 for metric in self.metrics_dict}

        self.classification_loss = nn.CrossEntropyLoss()
        
        if self.save_path is not None:
            os.makedirs(os.path.join(self.save_path, "last"), exist_ok=True)
            
    def get_postfix(self, loss, stage, log_metrics=False):
        
        postfix = {
            **({f"{stage} loss": f"{(loss):.3f}"} if stage == "train" else {}),
            **({metric_name: f"{metric.get_metric():.3f}" for metric_name, metric \
                in self.metrics_dict.items()} if (stage == "valid" and log_metrics) else {})}

        return postfix

    def log_to_wandb(self, step, loss, stage):
        
        metrics_to_log = {f"{stage}/step": step}
        
        if stage == "valid":
            for metric_name, metric in self.metrics_dict.items():
                metrics_to_log[f"{stage}/{metric_name}"] = metric.get_metric()
            
        if stage == "train" and self.lr_scheduler is not None:
            metrics_to_log[f"{stage}/{stage}_loss"] = loss
            metrics_to_log[f"{stage}/{stage}_lr"] = self.lr_scheduler.get_last_lr()[0]
        
        wandb.log(metrics_to_log)

    def compute_validation_metrics(self):
        for metric_name in self.metrics_dict:
            self.metrics_dict[metric_name].compute()

    def reset_metrics(self):
        for _, metric in self.metrics_dict.items():
            metric.reset()
        
    def process_batch(self, batch, stage="train"):
        batch["flattened_patches"] = batch["flattened_patches"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        
        if stage == "train":
            if self.cls_head:
                batch["target_type"] = batch["target_type"].to(self.device)
            
            batch["labels"] = batch["labels"].to(self.device)

        return batch

    def train_one_step(self, step, batch, total_steps):
        
        batch = self.process_batch(batch)

        flattened_patches = batch["flattened_patches"]
        attention_mask = batch["attention_mask"]
        target_types = batch["target_type"]
        labels = batch["labels"]
        
        with torch.cuda.amp.autocast(enabled=self.fp16):
            outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
        
        batch_loss = outputs["loss"] / self.grad_accum_steps
                      
        if self.cls_head:
            classification_loss = self.classification_loss(outputs["type_logits"], target_types)
            batch_loss += (classification_loss / self.grad_accum_steps)
        
        self.scaler.scale(batch_loss).backward()
        
        if (step % self.grad_accum_steps == 0) or (step == total_steps):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            scale = self.scaler.get_scale()
            self.scaler.update()

            if self.lr_scheduler is not None and (scale <= self.scaler.get_scale()):
                self.lr_scheduler.step()
                
            self.optimizer.zero_grad(set_to_none=True)
        
        return batch_loss.item()

    def train_one_epoch(self, epoch, train_dataloader, valid_dataloader):
        
        self.model.train()

        total_steps = len(train_dataloader) * (epoch)
        start_step = (epoch - 1) * len(train_dataloader) + 1
        total_train_loss = 0

        with tqdm(enumerate(train_dataloader, start_step), unit="batch", total=total_steps, bar_format='{l_bar}{bar:10}{r_bar}', position=0, leave=True, initial=start_step) as progress_bar:

            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                internal_step = (step + 1) - start_step
                
                total_train_loss += self.train_one_step(step, batch, total_steps=total_steps)
                self.checkpoint(step, total_train_loss, "train")
                
                if (internal_step % self.grad_accum_steps == 0) or (step == total_steps):
                    current_step = step // self.grad_accum_steps
                    current_loss = total_train_loss / internal_step
                    
                    progress_bar.set_postfix(self.get_postfix(current_loss, "train"))
                    
                    if self.wandb_log:
                        self.log_to_wandb(current_step, current_loss, "train")
                        
                if epoch >= self.first_eval_epoch:
                    self.validate_one_epoch(step, valid_dataloader)
                    self.model.train()
                
        total_train_loss /= total_steps
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return total_train_loss

    def validate_one_step(self, batch):
        
        batch = self.process_batch(batch)

        flattened_patches = batch["flattened_patches"]
        attention_mask = batch["attention_mask"]
        target_types = batch["target_type"]

        target_texts_x = batch["target_text_x"]
        target_texts_y = batch["target_text_y"]
        
        x_axis_types = batch["x_axis_types"]
        y_axis_types = batch["y_axis_types"]
        
        with torch.cuda.amp.autocast(enabled=self.fp16):
            if self.cls_head:
                predictions, types = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=self.max_token_len)
            else:
                predictions = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=self.max_token_len)
                
            predictions = self.processor.batch_decode(predictions, skip_special_tokens=True)

            if not self.cls_head:
                types = [TYPE2INT.get(pred.split("<0x0A>")[0], 5) for pred in predictions]
                predictions = ["<0x0A>".join(pred.split("<0x0A>")[1:]) for pred in predictions]
            
            predictions_x, predictions_y = process_output(predictions)
            
            for metric_name in self.metrics_dict:
                self.metrics_dict[metric_name].update(predictions_text=[*predictions_x, *predictions_y], labels_text=[*target_texts_x, *target_texts_y], \
                    prediction_type=[i for i in types for _ in range(2)], labels_type=[i.item() for i in target_types for _ in range(2)], axis_types=[*x_axis_types, *y_axis_types])
        
    @torch.no_grad()
    def validate_one_epoch(self, global_step, dataloader, force=False):
        if not force and not (global_step % self.validation_step) == 0 or (global_step == 0):
            return
        
        self.model.eval()
        
        total_valid_loss = 0
        total_steps = len(dataloader)

        with tqdm(enumerate(dataloader, 1), unit="batch", bar_format='{l_bar}{bar:10}{r_bar}', total=total_steps, position=0, leave=True) as progress_bar:
            progress_bar.set_description(f"Validation {global_step}".ljust(15))

            for step, batch in progress_bar:
                self.validate_one_step(batch)

                if step == total_steps:
                    self.compute_validation_metrics()

                progress_bar.set_postfix(
                    self.get_postfix(None, "valid", log_metrics=(step==total_steps)))

        total_valid_loss /= total_steps

        if self.wandb_log:
            self.log_to_wandb(global_step, total_valid_loss, "valid")
            
        self.checkpoint(global_step, total_valid_loss, "valid")
        self.reset_metrics()

        torch.cuda.empty_cache()
        gc.collect()
        
    def checkpoint(self, global_step, loss, stage):
        if self.save_path is None:
            return

        if global_step % 500 == 0 and stage == "train":

            torch.save(self.model.state_dict(), os.path.join(self.save_path, "last", "pytorch_model.pth"))

            torch.save({
                'loss': loss,
                'global_step': global_step,
                'scaler_state_dict': self.scaler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                }, os.path.join(self.save_path, "last", "training_states.pth"))

            return

        if stage == "train":
            return
        
        for metric_name, best_metric_value in self.best_metrics.items():
            matric_value = self.metrics_dict[metric_name].get_metric()
            
            if matric_value < best_metric_value:
                continue

            self.best_metrics[metric_name] = matric_value
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f"best_{metric_name}"))

    def train(self, epochs, train_dataloader, valid_dataloader):
        
        torch.cuda.empty_cache()
            
        for epoch in range(1, epochs+1):
            self.train_one_epoch(epoch, train_dataloader, valid_dataloader)

        self.validate_one_epoch(len(train_dataloader) * epochs, valid_dataloader, force=True)
        
        return self.best_metrics
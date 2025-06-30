from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup
import lightning as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from huggingface_hub import login
from dataclasses import dataclass
import torch
import torch.nn as nn
from data import get_loaders
import os

login(os.getenv('HF_TOKEN'))

hf_model_hub = "babs/warm-up"

@dataclass
class TrainingConfig:
    model_id: str = "Oluwadara/finetuned-whisper-asr-track-a"
    batch_size: int = 8
    max_epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0001
    save_top_k: int = 5
    monitor: str = "valid_loss"
    warmup_ratio: float = 0.1
    num_workers: int = 8

cfg = TrainingConfig()

class ASRTrainer(LightningModule):
    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.cfg = cfg
        self.train_loader, self.val_loader = get_loaders(cfg.batch_size)

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.learning_rate, fused=True, betas=(0.9, 0.999))

        total_steps = self.cfg.max_epochs *90_000
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def compute_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log_dict({
            "train_loss": loss,
            "grad_norm": self.compute_grad_norm()
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log_dict({
            "valid_loss": loss,
            "grad_norm": self.compute_grad_norm()
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        try:
            self.model.push_to_hub(hf_model_hub, commit_message="checkpoint", private=True)
        except Exception as e:
            print(f"Error pushing to HF hub: {e}")

logger = WandbLogger(
    project="whisper-asr-finetuning",
    name="finetune-whisper-asr",
    log_model=True,
    save_dir="./wandb_logs"
)

learning_rate_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step",
                                                                 log_momentum=True,
                                                                 log_weight_decay=True)

trainer = Trainer(
    max_epochs=cfg.max_epochs,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            dirpath="./checkpoints",
            filename="whisper-asr-{epoch:02d}-{val_loss:.2f}",
            monitor=cfg.monitor,
            mode="min",
            every_n_train_steps=15000,
            save_weights_only=False,
            save_top_k = 5
        )
    ],
    profiler="simple",
    accumulate_grad_batches=6,
    enable_progress_bar=True,
    precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "32",
    strategy="auto",
    log_every_n_steps=10,
    #gradient_clip_val=1.0,
    val_check_interval=3500
)

# Run training
model = ASRTrainer(cfg)
trainer.fit(model)

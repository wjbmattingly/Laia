import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import CTCLoss
from typing import Optional, Dict, Any
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ..utils.metrics import TextRecognitionMetrics

class CTCTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        char_map: Dict[str, int],
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        use_distortions: bool = False,
        grad_clip: float = 0.0,
        cer_trim: Optional[int] = None,
        optimizer_class: Any = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.ctc_loss = CTCLoss(zero_infinity=True)
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.metrics = TextRecognitionMetrics(char_map, cer_trim=cer_trim)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwargs
        )
        return optimizer
    
    def _compute_loss_and_metrics(self, batch, batch_idx, prefix=''):
        images, texts, input_lengths, target_lengths = batch
        
        # Forward pass
        log_probs = self(images)  # (B, T, C)
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=2)
        
        # CTC loss
        loss = self.ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, C)
            texts,
            input_lengths,
            target_lengths
        )
        
        # Get target texts for metrics
        target_texts = []
        for text, length in zip(texts, target_lengths):
            target_texts.append(''.join([self.metrics.idx_to_char[i.item()] for i in text[:length]]))
        
        # Compute metrics
        metrics = self.metrics(log_probs.transpose(0, 1), target_texts)
        
        # Log everything
        self.log(f'{prefix}loss', loss, prog_bar=True)
        self.log(f'{prefix}cer', metrics['cer'], prog_bar=True)
        self.log(f'{prefix}wer', metrics['wer'], prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss_and_metrics(batch, batch_idx, prefix='train_')
        
        if self.hparams.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self._compute_loss_and_metrics(batch, batch_idx, prefix='val_')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CTCTrainer")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--use_distortions", type=bool, default=False)
        parser.add_argument("--grad_clip", type=float, default=0.0)
        parser.add_argument("--cer_trim", type=int, default=None)
        return parent_parser 
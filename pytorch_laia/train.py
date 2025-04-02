import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from laia.models.crnn import CRNN
from laia.trainers.ctc_trainer import CTCTrainer
from laia.data.handwriting_dataset import HandwritingDataset
from laia.utils.image_distorter import ImageDistorter

def main():
    parser = argparse.ArgumentParser(description="Train handwritten text recognition model")
    
    # Add program level args
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--train_gt", type=str, required=True, help="Training ground truth file")
    parser.add_argument("--val_gt", type=str, required=True, help="Validation ground truth file")
    parser.add_argument("--char_map", type=str, required=True, help="Character map JSON file")
    parser.add_argument("--img_height", type=int, default=64, help="Input image height")
    parser.add_argument("--max_width", type=int, default=None, help="Max input image width")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--wandb_project", type=str, default="laia", help="Weights & Biases project name")
    
    # Add model specific args
    parser = CRNN.add_model_specific_args(parser)
    parser = CTCTrainer.add_model_specific_args(parser)
    parser = ImageDistorter.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # Load character map
    with open(args.char_map, 'r', encoding='utf-8') as f:
        char_map = json.load(f)
    
    # Create datasets
    train_transform = ImageDistorter(
        max_rotation=args.max_rotation,
        max_scale=args.max_scale,
        max_shear=args.max_shear,
        max_translation=args.max_translation,
        random_elastic=args.random_elastic,
        elastic_sigma=args.elastic_sigma,
        elastic_alpha=args.elastic_alpha
    ) if args.use_distortions else None
    
    train_dataset = HandwritingDataset(
        args.data_dir,
        args.train_gt,
        char_map,
        transform=train_transform,
        img_height=args.img_height,
        max_width=args.max_width
    )
    
    val_dataset = HandwritingDataset(
        args.data_dir,
        args.val_gt,
        char_map,
        img_height=args.img_height,
        max_width=args.max_width
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=HandwritingDataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=HandwritingDataset.collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = CRNN(
        num_classes=len(char_map),
        cnn_output_size=args.cnn_output_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = CTCTrainer(
        model=model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_distortions=args.use_distortions,
        grad_clip=args.grad_clip
    )
    
    # Setup training
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='laia-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    wandb_logger = WandbLogger(project=args.wandb_project)
    
    # Create PyTorch Lightning trainer
    pl_trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=16  # Use mixed precision for faster training
    )
    
    # Train
    pl_trainer.fit(trainer, train_loader, val_loader)

if __name__ == "__main__":
    main() 
# PyTorch Laia

This is a PyTorch 2.0+ port of the Laia framework for handwritten text recognition and document analysis. The original Torch7/Lua implementation has been modernized to take advantage of PyTorch's latest features and best practices.

## Key Features

- Modern PyTorch 2.0+ implementation
- PyTorch Lightning integration for structured training
- Improved CTC (Connectionist Temporal Classification) training
- Enhanced image distortion and augmentation
- Better experiment tracking with Weights & Biases
- Type hints and modern Python practices

## Installation

1. Create a new Python environment:
```bash
conda create -n pytorch_laia python=3.10
conda activate pytorch_laia
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
pytorch_laia/
├── laia/
│   ├── models/        # Neural network model definitions
│   ├── trainers/      # Training components (CTC, etc.)
│   ├── utils/         # Utilities (image distortion, etc.)
│   └── nn/           # Neural network modules
├── requirements.txt
└── README.md
```

## Migration Notes

This is a complete rewrite of the original Lua/Torch7 implementation in modern PyTorch. Key changes include:

1. **CTC Training**: Uses PyTorch's native `CTCLoss` instead of warp-ctc
2. **Training Framework**: Uses PyTorch Lightning for structured training loops
3. **Image Distortions**: Modernized implementation using PyTorch's transforms
4. **Experiment Tracking**: Integration with Weights & Biases
5. **Type Safety**: Added type hints throughout the codebase

## Usage Example

```python
import pytorch_lightning as pl
from laia.trainers import CTCTrainer
from laia.utils import ImageDistorter
from your_model import YourModel

# Create model and trainer
model = YourModel()
trainer = CTCTrainer(
    model=model,
    learning_rate=1e-3,
    batch_size=16,
    use_distortions=True
)

# Create PyTorch Lightning trainer
pl_trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    logger=WandbLogger(project='laia')
)

# Train
pl_trainer.fit(trainer, train_dataloader, val_dataloader)
```

```bash
python train.py \
    --data_dir /path/to/images \
    --train_gt train.json \
    --val_gt val.json \
    --char_map char_map.json \
    --batch_size 32 \
    --max_epochs 100 \
    --use_distortions True
```


```bash
python evaluate.py \
    --data_dir /path/to/images \
    --gt_file test.json \
    --char_map char_map.json \
    --checkpoint path/to/checkpoint.ckpt \
    --gpu \
    --output_file predictions.json
```
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project maintains the same license as the original Laia project. 
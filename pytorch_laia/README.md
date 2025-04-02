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


The system expects two main components:

1. **Image Files**:
   - Handwritten text images in a directory
   - Common image formats (jpg, png, etc.)
   - Images will be automatically:
     - Converted to grayscale
     - Resized to a fixed height (default 64px) while maintaining aspect ratio
     - Normalized to [0, 1] range

2. **Ground Truth Files** (JSON format):
   - Separate files for training, validation, and testing
   - Each file should be a JSON list of objects with this structure:
```json
[
    {
        "image": "path/to/image1.png",
        "text": "transcription of the text"
    },
    {
        "image": "path/to/image2.png",
        "text": "another transcription"
    },
    ...
]
```

3. **Character Map File** (JSON format):
   - Maps characters to integer indices
   - Must include all characters that appear in your transcriptions
   - Index 0 is reserved for the CTC blank token
   - Example:
```json
{
    "<blank>": 0,
    " ": 1,
    "a": 2,
    "b": 3,
    "c": 4,
    ...
}
```

Here's a complete example of the directory structure:

```
data/
├── images/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── char_map.json
├── train.json
├── val.json
└── test.json
```

To use this with the training script:

```bash
python train.py \
    --data_dir data/images \
    --train_gt data/train.json \
    --val_gt data/val.json \
    --char_map data/char_map.json \
    --img_height 64 \
    --batch_size 32
```

For evaluation:

```bash
python evaluate.py \
    --data_dir data/images \
    --gt_file data/test.json \
    --char_map data/char_map.json \
    --checkpoint path/to/model.ckpt \
    --output_file predictions.json
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project maintains the same license as the original Laia project. 
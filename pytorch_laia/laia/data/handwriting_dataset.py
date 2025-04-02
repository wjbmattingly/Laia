import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

class HandwritingDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        gt_file: str,
        char_map: Dict[str, int],
        transform=None,
        img_height: int = 64,
        max_width: Optional[int] = None
    ):
        """
        Dataset for handwritten text recognition.
        
        Args:
            data_dir: Directory containing the images
            gt_file: Path to ground truth file (JSON format with "image" and "text" fields)
            char_map: Dictionary mapping characters to indices
            transform: Optional transform to be applied to images
            img_height: Height to resize images to (maintaining aspect ratio)
            max_width: Maximum width of images after resizing (None for no limit)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_height = img_height
        self.max_width = max_width
        self.char_map = char_map
        
        # Load ground truth
        with open(gt_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text string to tensor of character indices."""
        return torch.tensor([self.char_map.get(c, 0) for c in text], dtype=torch.long)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            text: Tensor of encoded characters
            width: Original width of image (used for CTC input length)
        """
        sample = self.samples[idx]
        img_path = self.data_dir / sample["image"]
        
        # Load and convert to grayscale
        img = Image.open(img_path).convert('L')
        width = img.width
        
        # Resize maintaining aspect ratio
        ratio = self.img_height / img.height
        new_width = int(width * ratio)
        if self.max_width:
            new_width = min(new_width, self.max_width)
        img = img.resize((new_width, self.img_height), Image.Resampling.BILINEAR)
        
        # Convert to tensor
        img = torch.FloatTensor(list(img.getdata())).view(1, self.img_height, new_width)
        img = img / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            img = self.transform(img)
            
        # Encode text
        text = self.encode_text(sample["text"])
        
        return img, text, new_width
        
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        """Custom collate function for DataLoader that handles variable width images."""
        # Sort by width for more efficient batching
        batch.sort(key=lambda x: x[2], reverse=True)
        images, texts, widths = zip(*batch)
        
        # Pad images to max width in batch
        max_width = max(widths)
        batch_size = len(images)
        height = images[0].size(1)
        padded_images = torch.zeros(batch_size, 1, height, max_width)
        
        for i, img in enumerate(images):
            padded_images[i, :, :, :img.size(2)] = img
            
        # Prepare for CTC loss
        text_lengths = torch.tensor([len(t) for t in texts])
        max_text_length = text_lengths.max()
        padded_texts = torch.zeros(batch_size, max_text_length, dtype=torch.long)
        
        for i, text in enumerate(texts):
            padded_texts[i, :len(text)] = text
            
        # Calculate input lengths for CTC (assuming model reduces width by factor of 4)
        input_lengths = torch.tensor([w // 4 for w in widths])
        
        return padded_images, padded_texts, input_lengths, text_lengths 
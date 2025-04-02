import argparse
import json
from pathlib import Path
import random
from PIL import Image
from typing import Dict, List, Set
import os

def collect_characters(transcriptions: List[str]) -> Set[str]:
    """Collect all unique characters from transcriptions."""
    chars = set()
    for text in transcriptions:
        chars.update(set(text))
    return chars

def create_char_map(chars: Set[str], output_file: str):
    """Create character map file."""
    # Start with blank token
    char_map = {"<blank>": 0}
    
    # Add space first (if present)
    if " " in chars:
        char_map[" "] = len(char_map)
        chars.remove(" ")
    
    # Add remaining characters
    for c in sorted(chars):
        char_map[c] = len(char_map)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(char_map, f, indent=4, ensure_ascii=False)
    
    return char_map

def create_gt_file(data: List[Dict], output_file: str):
    """Create ground truth JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Prepare data for PyTorch Laia")
    parser.add_argument("--images_dir", type=str, required=True,
                      help="Directory containing image files")
    parser.add_argument("--transcription_file", type=str, required=True,
                      help="File containing image paths and transcriptions (one per line, tab-separated)")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for prepared data")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                      help="Fraction of data to use for testing")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for splitting data")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read transcription file
    print("Reading transcription file...")
    data = []
    chars = set()
    with open(args.transcription_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                img_path, text = line.strip().split('\t')
                # Verify image exists and can be opened
                img_file = Path(args.images_dir) / img_path
                try:
                    img = Image.open(img_file)
                    img.close()
                    data.append({
                        "image": img_path,
                        "text": text
                    })
                    chars.update(set(text))
                except Exception as e:
                    print(f"Warning: Could not open image {img_file}: {e}")
    
    print(f"Found {len(data)} valid image-text pairs")
    print(f"Found {len(chars)} unique characters")
    
    # Create character map
    print("\nCreating character map...")
    char_map = create_char_map(chars, output_dir / "char_map.json")
    print(f"Character map saved to {output_dir / 'char_map.json'}")
    
    # Split data
    random.seed(args.random_seed)
    random.shuffle(data)
    
    val_size = int(len(data) * args.val_split)
    test_size = int(len(data) * args.test_split)
    train_size = len(data) - val_size - test_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    print("\nSaving data splits...")
    create_gt_file(train_data, output_dir / "train.json")
    create_gt_file(val_data, output_dir / "val.json")
    create_gt_file(test_data, output_dir / "test.json")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nData preparation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    main() 
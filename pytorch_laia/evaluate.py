import argparse
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from laia.models.crnn import CRNN
from laia.data.handwriting_dataset import HandwritingDataset
from laia.utils.metrics import TextRecognitionMetrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate handwritten text recognition model")
    
    # Add program level args
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth file")
    parser.add_argument("--char_map", type=str, required=True, help="Character map JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate")
    parser.add_argument("--img_height", type=int, default=64, help="Input image height")
    parser.add_argument("--max_width", type=int, default=None, help="Max input image width")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--cer_trim", type=int, default=None, help="Character index to trim for CER calculation")
    parser.add_argument("--output_file", type=str, help="Save predictions to file")
    
    # Add model specific args
    parser = CRNN.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # Load character map
    with open(args.char_map, 'r', encoding='utf-8') as f:
        char_map = json.load(f)
    
    # Create dataset and loader
    dataset = HandwritingDataset(
        args.data_dir,
        args.gt_file,
        char_map,
        img_height=args.img_height,
        max_width=args.max_width
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=HandwritingDataset.collate_fn,
        pin_memory=True
    )
    
    # Create model and load checkpoint
    model = CRNN(
        num_classes=len(char_map),
        cnn_output_size=args.cnn_output_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=0.0  # No dropout during evaluation
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists (from Lightning checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Setup metrics
    metrics = TextRecognitionMetrics(char_map, cer_trim=args.cer_trim)
    
    # Evaluate
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, texts, input_lengths, text_lengths = batch
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = torch.nn.functional.log_softmax(outputs, dim=2)
            outputs = outputs.transpose(0, 1)  # (T, B, C)
            
            # Get target texts
            target_texts = []
            for text, length in zip(texts, text_lengths):
                target_texts.append(''.join([metrics.idx_to_char[i.item()] for i in text[:length]]))
            
            # Decode predictions
            predictions = metrics.decode_predictions(outputs)
            
            all_predictions.extend(predictions)
            all_targets.extend(target_texts)
    
    # Compute metrics
    results = metrics(outputs, all_targets)  # Just for the last batch metrics
    total_cer = metrics.compute_cer(all_predictions, all_targets)
    total_wer = metrics.compute_wer(all_predictions, all_targets)
    
    print(f"\nResults:")
    print(f"Character Error Rate: {total_cer:.4f}")
    print(f"Word Error Rate: {total_wer:.4f}")
    
    # Save predictions if requested
    if args.output_file:
        output_data = []
        for pred, target in zip(all_predictions, all_targets):
            output_data.append({
                'prediction': pred,
                'target': target
            })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nPredictions saved to {args.output_file}")

if __name__ == "__main__":
    main() 
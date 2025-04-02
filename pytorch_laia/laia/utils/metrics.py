import torch
import editdistance
from typing import List, Dict, Optional

class TextRecognitionMetrics:
    def __init__(self, char_map: Dict[str, int], cer_trim: Optional[int] = None):
        """
        Metrics for text recognition evaluation.
        
        Args:
            char_map: Dictionary mapping characters to indices
            cer_trim: If not None, removes leading, trailing and repetitions 
                     of the character with this index before computing CER
        """
        self.char_map = char_map
        self.idx_to_char = {v: k for k, v in char_map.items()}
        self.cer_trim = cer_trim
        
    def decode_predictions(self, log_probs: torch.Tensor) -> List[str]:
        """
        Decode model predictions using greedy decoding.
        
        Args:
            log_probs: Tensor of shape (T, B, C) containing log probabilities
            
        Returns:
            List of decoded strings
        """
        # Get most likely character at each timestep
        predictions = log_probs.argmax(dim=-1)  # (T, B)
        predictions = predictions.transpose(0, 1)  # (B, T)
        
        decoded = []
        for pred in predictions:
            # Remove repeated characters
            collapsed = []
            prev_char = None
            for p in pred:
                if p != prev_char:  # Skip repeats
                    collapsed.append(p.item())
                    prev_char = p
                    
            # Remove blanks (index 0)
            collapsed = [p for p in collapsed if p != 0]
            
            # Convert to string
            text = ''.join([self.idx_to_char[p] for p in collapsed])
            decoded.append(text)
            
        return decoded
        
    def compute_cer(self, predictions: List[str], targets: List[str]) -> float:
        """Compute Character Error Rate."""
        total_chars = 0
        total_dist = 0
        
        for pred, target in zip(predictions, targets):
            if self.cer_trim is not None:
                # Remove leading/trailing spaces and collapse repeated spaces
                trim_char = self.idx_to_char[self.cer_trim]
                pred = ' '.join(pred.split(trim_char)).strip()
                target = ' '.join(target.split(trim_char)).strip()
                
            dist = editdistance.eval(pred, target)
            total_dist += dist
            total_chars += len(target)
            
        return total_dist / total_chars if total_chars > 0 else 1.0
        
    def compute_wer(self, predictions: List[str], targets: List[str]) -> float:
        """Compute Word Error Rate."""
        total_words = 0
        total_dist = 0
        
        for pred, target in zip(predictions, targets):
            pred_words = pred.split()
            target_words = target.split()
            
            dist = editdistance.eval(pred_words, target_words)
            total_dist += dist
            total_words += len(target_words)
            
        return total_dist / total_words if total_words > 0 else 1.0
        
    def __call__(self, log_probs: torch.Tensor, targets: List[str]) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            log_probs: Tensor of shape (T, B, C) containing log probabilities
            targets: List of target strings
            
        Returns:
            Dictionary containing CER and WER
        """
        predictions = self.decode_predictions(log_probs)
        return {
            'cer': self.compute_cer(predictions, targets),
            'wer': self.compute_wer(predictions, targets)
        } 
import torch
import torch.nn as nn
from typing import Optional, Tuple

class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        cnn_output_size: int = 512,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Convolutional Recurrent Neural Network for handwritten text recognition.
        
        Args:
            num_classes: Number of output classes (including blank)
            input_channels: Number of input channels (1 for grayscale)
            cnn_output_size: Size of CNN feature maps
            lstm_hidden_size: Number of LSTM hidden units
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 x H/2 x W/2
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 x H/4 x W/4
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256 x H/8 x W/4
            
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 6
            nn.Conv2d(512, cnn_output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512 x H/16 x W/4
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=cnn_output_size * 4,  # Height is reduced to H/16
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction: (B, C, H, W) -> (B, C', H', W')
        conv = self.cnn(x)
        
        # Prepare for RNN: (B, C', H', W') -> (B, W', C'*H')
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (B, W, C, H)
        conv = conv.reshape(batch, width, channels * height)
        
        # RNN sequence modeling: (B, W, C'*H') -> (B, W, 2*H)
        rnn, _ = self.rnn(conv)
        
        # Classification: (B, W, 2*H) -> (B, W, num_classes)
        output = self.classifier(rnn)
        
        return output
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CRNN")
        parser.add_argument("--cnn_output_size", type=int, default=512)
        parser.add_argument("--lstm_hidden_size", type=int, default=256)
        parser.add_argument("--lstm_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.5)
        return parent_parser 
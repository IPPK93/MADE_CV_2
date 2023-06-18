import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 180) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 512
        self.height = 32
        self.rnn_hidden_size = 256
        self.num_classes = num_classes
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=2, stride=1, padding=0),
        )
                
        self.rec_layers = nn.LSTM(self.out_channels, self.rnn_hidden_size, bidirectional=True, num_layers=2)
        
        self.clf_layer = nn.Linear(2 * self.rnn_hidden_size, self.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # Outputs (Bsize, Cnum, H, W)
        
        batch_size, ch_num, height, width = x.size()
                
        # We need to concat all channels columnwise
        x = x.view(batch_size, ch_num * height, width)
        x = x.permute(2, 0, 1)

        x, _ = self.rec_layers(x)
        x = self.clf_layer(x)
        return x  # Output is of shape (width, bsize, num_classes)
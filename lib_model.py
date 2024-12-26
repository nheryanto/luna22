import torch.nn as nn
import timm_3d
from timm_3d.layers.classifier import ClassifierHead
    
class SingleTaskNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_class: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super(SingleTaskNet, self).__init__()
        model = timm_3d.create_model(model_name, pretrained)
        num_features = model.get_classifier().in_features
        if 'dense' in model_name: self.encoder = nn.Sequential(*list(model.children())[:-3])
        else: self.encoder = nn.Sequential(*list(model.children())[:-2])
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)
        self.drop = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(num_features, 512, bias=True)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(512, n_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

import torch
class SkipResNet(nn.Module):
    def __init__(self, in_channels=3, n_class=1):
        super(SkipResNet, self).__init__()
        self.in_channels = in_channels
        self.n_class = n_class

        # first branch with 3x3x3 convolution
        self.CBR_init1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),  # 3x3x3 convolution
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # second branch with 1x1x1 convolution
        self.CBR_init2 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=1),  # 1x1x1 convolution
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # first skip connected structure
        self.down_sample1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=2, stride=2),  # 3x3x3 convolution
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.CBR1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),  # 3x3x3 convolution
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # second skip connected structure
        self.down_sample2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=2, stride=2),  # 3x3x3 convolution
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.CBR2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),  # 3x3x3 convolution
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # third skip connected structure
        self.down_sample3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=2, stride=2),  # 3x3x3 convolution
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.CBR3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  # 3x3x3 convolution
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # fourth skip connected structure
        self.down_sample4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=2, stride=2),  # 3x3x3 convolution
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.CBR4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),  # 3x3x3 convolution
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(256, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, self.n_class, bias=True)

    def forward(self, x):
        x0_1 = self.CBR_init1(x)
        x0_2 = self.CBR_init2(x)
        x0 = torch.cat((x0_1, x0_2), dim=1)
        
        # first skip connected structure
        x1_1 = self.down_sample1(x0)
        x1_2 = self.CBR1(x1_1)
        x1_2 = self.CBR1(x1_2)
        x1_3 = x1_1 + x1_2 # residual
        x1 = torch.cat((x1_1, x1_3), dim=1)
        
        # second skip connected structure
        x2_1 = self.down_sample2(x1)
        x2_2 = self.CBR2(x2_1)
        x2_2 = self.CBR2(x2_2)
        x2_2 = self.CBR2(x2_2)
        x2_3 = x2_1 + x2_2 # residual
        x2 = torch.cat((x2_1, x2_3), dim=1)
        
        # third skip connected structure
        x3_1 = self.down_sample3(x2)
        x3_2 = self.CBR3(x3_1)
        x3_2 = self.CBR3(x3_2)
        x3_2 = self.CBR3(x3_2)
        x3_3 = x3_1 + x3_2 # residual
        x3 = torch.cat((x3_1, x3_3), dim=1)
        
        # fourth skip connected structure
        x4_1 = self.down_sample4(x3)
        x4_2 = self.CBR4(x4_1)
        x4_2 = self.CBR4(x4_2)
        x4_2 = self.CBR4(x4_2)
        x4_3 = x4_1 + x4_2 # residual
        x4 = torch.cat((x4_1, x4_3), dim=1)
        
        x5 = self.pool(x4)
        x5 = self.flatten(x5)
        x6 = self.fc(x5)
        x6 = self.relu(x6)
        x7 = self.fc1(x6)

        return x7
        
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv3d):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm3d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
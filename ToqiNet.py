"""
Hello everyone, Hope you are enjoying the day . As you have access the model. Let me tell you that this model is in development rigt now
its not the final version. Its inspired by the AlexNet but it has larger parametar then AlexNet although the accurecy is not decent 
right now. Still its much more simplier then any other model available. This model also provide you the Coustomdataset 
feature which implies to autoresize and image size handeling while training image.

COPYRIGHT RESERVED BY : thameedtoqi123@gmail.com (C) 2024


"""

import torch
import torch.nn as nn


class ToqiNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ToqiNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 8192),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8192, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

input_shape = (3, 227, 227)  
num_classes = 2  

model = ToqiNet(num_classes)

print(model)

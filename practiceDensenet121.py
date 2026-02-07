import torch
import torch.nn as nn
from torchsummary import summary

def conv(in_channels,out_channels,kernel_size,stride, padding):
    return nn.Sequential(
    nn.BatchNorm2d(in_channels),
    nn.ReLU(),
    nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding)
    )    

    
def convAvgPool(in_channels,out_channels,kernel_size,pool_kernel_size,conv_stride,pool_stride, padding,drop_rate=0.2):
    return nn.Sequential(
        conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=conv_stride, padding=padding),
        nn.Dropout2d(drop_rate),
        nn.AvgPool2d(kernel_size=pool_kernel_size,stride=pool_stride, padding=padding)
    )


class dense(nn.Module):
    def __init__(self,num_layers,in_channels,out_channels,k=32):
        super().__init__()
        self.num_layers=num_layers
        self.conv1=nn.ModuleList()
        self.conv2=nn.ModuleList()
        for i in range(num_layers):
            self.conv1.append(conv(in_channels=in_channels+k*i,out_channels=4*k,kernel_size=1,stride=1, padding=0))
            self.conv2.append(conv(in_channels=4*k,out_channels=k,kernel_size=3,stride=1, padding=1))

    def forward(self,x):
        all_features=[x]
        for i in range(self.num_layers):
            current_input=torch.cat(all_features, dim=1)
            y=self.conv1[i](current_input)
            y=self.conv2[i](y)
            all_features.append(y)
            
        return torch.cat(all_features, dim=1)
                
def makeDensenet121(k=32):
    return nn.Sequential(
    nn.Conv2d(in_channels=(n:=3),out_channels=(n:=2*k),kernel_size=7,stride=2, padding=3),
    nn.BatchNorm2d(n),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
    #grayscale images

    dense(num_layers=6,in_channels=n,out_channels=(n:=n+6*k)),

    convAvgPool(in_channels=n,out_channels=(n:=n//2),kernel_size=1,pool_kernel_size=2,conv_stride=1,pool_stride=2, padding=0),

    dense(num_layers=12,in_channels=n,out_channels=(n:=n+12*k)),

    convAvgPool(in_channels=n,out_channels=(n:=n//2),kernel_size=1,pool_kernel_size=2,conv_stride=1,pool_stride=2, padding=0),
    
    dense(num_layers=24,in_channels=n,out_channels=(n:=n+24*k)),
    
    convAvgPool(in_channels=n,out_channels=(n:=n//2),kernel_size=1,pool_kernel_size=2,conv_stride=1,pool_stride=2, padding=0),
    
    dense(num_layers=16,in_channels=n,out_channels=(n:=n+16*k)),

    nn.AdaptiveAvgPool2d(1),
    
    nn.Flatten(),
    
    nn.Linear(n,1000)
    )

model=makeDensenet121()

summary(model, (3, 224, 224))

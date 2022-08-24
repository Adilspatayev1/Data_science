from torch import nn
#  Define the model

class AudioClassifier(nn.Module):
    def conv_block(self, in_feat, out_feat, kernel_size, stride, padding):
        # Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        conv_layer = []
        conv = nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding)
        relu = nn.ReLU()
        bn = nn.BatchNorm2d(out_feat)
        nn.init.kaiming_normal_(conv.weight, a=0.1)
        conv.bias.data.zero_()
        conv_layer += [conv, relu, bn]
        return conv_layer

    def __init__(self):
        super().__init__()    
        block1 = self.conv_block(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        block2 = self.conv_block(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        block3 = self.conv_block(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        block4 = self.conv_block(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        conv_layers = block1 + block2 + block3 + block4
        
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=14)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x) # adaptive pool
        x = x.view(x.shape[0], -1) # flatten
        x = self.lin(x) # linear layer
        return x
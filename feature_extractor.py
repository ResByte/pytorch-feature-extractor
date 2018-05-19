
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision import models 
import time
import argparse

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class FeatureExtractor(nn.Module):
    """Feature Extractor outputs convolutional features 
       using the imagenet pre-trained models. 
    """
    def __init__(self, arch = 'resnet18', pretrained=True):
        super(FeatureExtractor, self).__init__()

        model = models.__dict__[arch](pretrained=pretrained)
        # remove the classification layer 
        # this doesn't work with inception-v3 model 
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x)


parser = argparse.ArgumentParser(description='PyTorch FeatureExtractor')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--shape', '-s', default=224, type=int)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    
    

def main():
    args = parser.parse_args()
    
    # create feature extractor
    model = FeatureExtractor(args.arch, args.pretrained)

    # dummy input variable of order (N,C,H,W)
    x = torch.randn([1, 3, args.shape, args.shape ])
    x_var = Variable(x)

    # forward pass 
    s = time.time()
    y = model(x_var)
    e = time.time()
    print('Inference time:',e-s)
    
    # since this is on cpu we dont need to extract 
    print(y.squeeze().shape)


if __name__ == '__main__':
    main()
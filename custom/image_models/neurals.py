import math
import torch
import torch.nn as nn

class Sampler(nn.Module):
    def __init__(self, image_size, interpolate):
        super(Sampler, self).__init__()
        self.image_size = image_size
        self.interpolate = interpolate

    def forward(self, image):
        with torch.no_grad():
            return nn.functional.interpolate(
                input = image, 
                size = self.image_size, 
                mode = self.interpolate
            )
        

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class NewGeLU(nn.Module):
    def __init__(self):
        super(NewGeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
actv = {
    "gelu": GeLU(),
    "gelu_new": NewGeLU(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "linear": nn.Identity(),
}
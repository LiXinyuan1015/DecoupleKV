import torch
import torch.nn as nn
from image_models.loss import CELoss

BASIC_SIZE = 224

class ConvNetConfig(object):
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

example_config = ConvNetConfig({
    "image_size": 28,
    "image_channels": 1, 
    "num_labels": 10,
    "dropout_rate": 0.5, 
    "batchnorm_eps": 1e-12,
    "initializer_range": 2e-2,
    "activation": "relu",
    "interpolate": "bicubic",
    "conv_name": "alex",
    "device": "cuda",
})

class AlexNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        scale = BASIC_SIZE // self.config.image_size
        
        # assert scale * input_size == size0
        
        self.extract_feature = nn.Sequential(
            nn.Upsample(scale_factor = scale, mode = 'nearest'),
            nn.Conv2d(self.config.image_channels, 48 * 2, (11, 11), stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48 * 2, 128 * 2, (5, 5), padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128 * 2, 192* 2, (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(192 * 2, 192* 2, (3, 3), padding = 1),
            nn.ReLU(),
            nn.Conv2d(192 * 2, 128* 2, (3, 3), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(4096, self.config.num_labels)
        )
        
    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(feature.shape[0], -1)
        logits = self.linear(feature)
        return logits
    
conv_dict = {
    "alex": AlexNet
}
    
class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_model = conv_dict[config.conv_name](config)
        self.loss_func = CELoss()
        
        self.weight_init()
        self.to(config.device)
        
    def weight_init(self):
        for _, p in self.named_parameters():
            nn.init.trunc_normal_(p, std = self.conv_model.config.initializer_range)
        
    def forward(self, input_feature, label):
        logits = self.conv_model(input_feature)
        
        label_predict = logits.max(1)[1]
        model_output = (logits, label_predict)

        if label is not None:
            loss = self.loss_func(logits, label)
            model_output += (loss,)
        else:
            model_output += (None,)
        
        return model_output
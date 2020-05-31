# cnn_shape
Utility functions for calculating shapes of layer outputs in PyTorch. Can process and entire CNN and find the output shape of each consecutive layer. Produces a dictionary with layer names as keys and output shapes as values.


### Basic Usage

```python
import torch
import torch.nn as nn

import cnn_shape


class TestCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoder_pool1 = nn.MaxPool2d(2, 2)
        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder_tconv1 = nn.ConvTranspose2d(4, 16, 3, stride=1, padding=1)
        

network = TestCnn()
input = torch.ones((64, 3, 256, 256))
shapes = cnn_shape.get_layer_output_shapes(input.shape, network)
print(shapes)
```

```{'encoder_conv1': torch.Size([64, 16, 256, 256]), 'encoder_pool1': torch.Size([64, 16, 128, 128]), 'decoder_upsample1': torch.Size([64, 16, 256, 256]), 'decoder_tconv1': torch.Size([64, 16, 256, 256])}```

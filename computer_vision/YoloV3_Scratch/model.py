import torch
import torch.nn as nn

# Tuple : (out_channels, kernel_size, stride)
# List: ["B", 1] Blocks, no. of repeats
# B: Residual Block
# S: Scaled prediction Block & compute Yolo Loss
# U: Upsampling the feature map & concat with prev layer

#tuple, List, string
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (256, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4], # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",                   # Branch or stride 13x13 
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",                   # Branch or stride 26x26
    (256, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",                   # Branch or stride 52x52
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias= not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):  #B in config
    def __init__(self, channels, use_residual = True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                CNNBlock(channels, channels//2, kernel_size=1),
                CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels,3 *(num_classes + 5) , bn_act= False, kernel_size=1) #[prob_Obj, x, y, w, h] for 5 & 3 for 3_bbox for each cell 
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        # N x 3(Anchor Box) x 13 x 13(scaled_pred) x 5+num_classes (prob, x, y, w, h+num_classes)

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_new_layers()

    def forward(self, x):
        outputs = []
        route_connections = []  #skip connections & ConCat 

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue          # continue from the root of Scaled Prediction Block
                
            x = laer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x= torch.cat([x, route_connections[-1]], dim =1)
                route_connections.pop()

        return outputs


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels, 
                        out_channels,
                        kernel_size= kernel_size,
                        stride = stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels,use_residual= False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes = self.num_classes)
                    ]
                    in_channels = in_channels // 2 #continue from  last branch root

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers    


### Test case ###

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416 #Yolov1 :448, Yolov3: 416
    model = YOLOv3(num_classes = num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes+5)      #13 x 13
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes+5)      #26 x 26
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes +5)       #52 x 52 
    print("Success")























































































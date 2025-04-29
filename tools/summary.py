import torch
from thop import clever_format, profile
from torchsummary import summary
from nets.yolo import MSFNet_Head

if __name__ == "__main__":
    input_shape     = [416, 416]
    anchors_mask    = [[3, 4, 5], [1, 2, 3]]
    num_classes     = 5
    phi             = 0

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = MSFNet_Head(anchors_mask, num_classes, phi=phi).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))


import torch
import torch.nn as nn
from exercise03_template import CNN
from exercise05_ResNet_template import ResNet
from vgg11 import VGG11_BN 

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Compute MACs for convolution layer
def conv_macs(conv, H, W):
    Cin = conv.in_channels
    Cout = conv.out_channels
    K = conv.kernel_size[0]
    S = conv.stride[0]
    P = conv.padding[0]

    # output size
    Hout = (H + 2*P - K) // S + 1
    Wout = (W + 2*P - K) // S + 1

    macs = Hout * Wout * Cout * (Cin * K * K)
    return macs, Hout, Wout

# Compute MACs for ConvNet
def compute_cnn_macs():
    model = CNN()
    H = W = 32
    macs_total = 0

    # conv0
    macs, H, W = conv_macs(model.conv0, H, W)
    macs_total += macs
    # conv1
    macs, H, W = conv_macs(model.conv1, H, W)
    macs_total += macs
    # conv2
    macs, H, W = conv_macs(model.conv2, H, W)
    macs_total += macs

    # linear1 (Input: 128*12*12, Output: 128)
    macs_total += (128 * 12 * 12) * 128
    # linear2 (Input: 128, Output: 10)
    macs_total += 128 * 10

    return macs_total

# Compute MACs for ResNet
def compute_resnet_macs():
    model = ResNet()
    H = W = 32
    macs_total = 0
    
    # Initial Conv
    macs, H, W = conv_macs(model.conv1, H, W)
    macs_total += macs

    def block_macs(block, H, W):
        total = 0
        # conv1
        m1, H1, W1 = conv_macs(block.conv1, H, W)
        total += m1
        # conv2
        m2, H2, W2 = conv_macs(block.conv2, H1, W1)
        total += m2

        if block.downsample is not None:
            # downsample layer is usually index 0 in Sequential
            ds = block.downsample[0] 
            m3, _, _ = conv_macs(ds, H, W)
            total += m3

        return total, H2, W2

    # Iterate through blocks
    for blk in [model.block1_1, model.block1_2, model.block1_3]:
        m, H, W = block_macs(blk, H, W)
        macs_total += m
    for blk in [model.block2_1, model.block2_2, model.block2_3]:
        m, H, W = block_macs(blk, H, W)
        macs_total += m
    for blk in [model.block3_1, model.block3_2, model.block3_3]:
        m, H, W = block_macs(blk, H, W)
        macs_total += m

    # FC layer
    macs_total += 128 * 10

    return macs_total

# ==========================================
# New: Compute MACs for VGG11
# ==========================================
def compute_vgg_macs():
    model = VGG11_BN()
    H = W = 32
    macs_total = 0
    
    # VGG11 features is a Sequential container
    # We iterate through it and calculate MACs for Conv2d layers
    # We also need to update H, W for MaxPool2d layers
    
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            macs, H, W = conv_macs(layer, H, W)
            macs_total += macs
        elif isinstance(layer, nn.MaxPool2d):
            # Update spatial dimensions
            # Usually kernel=2, stride=2
            K = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            S = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            P = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            H = (H + 2*P - K) // S + 1
            W = (W + 2*P - K) // S + 1
    
    # Classifier layers (Linear)
    # Based on VGG11_BN definition:
    # 1. Linear 512*1*1 -> 4096
    macs_total += (512 * 1 * 1) * 4096
    # 2. Linear 4096 -> 4096
    macs_total += 4096 * 4096
    # 3. Linear 4096 -> 10
    macs_total += 4096 * 10
    
    return macs_total

def main():
    print("\n == CNN ==")
    cnn = CNN()
    print(f"Parameters: {count_parameters(cnn):,}")
    print(f"MACs: {compute_cnn_macs():,}")

    print("\n == VGG11 ==")
    vgg = VGG11_BN()
    print(f"Parameters: {count_parameters(vgg):,}")
    print(f"MACs: {compute_vgg_macs():,}")

    print("\n == ResNet ==")
    res = ResNet()
    print(f"Parameters: {count_parameters(res):,}")
    print(f"MACs: {compute_resnet_macs():,}")

if __name__ == "__main__":
    main()

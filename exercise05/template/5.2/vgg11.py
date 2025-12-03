import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
import os

# ==========================================
# VGG11 with BatchNorm Definition
# ==========================================
class VGG11_BN(nn.Module):
    def __init__(self):
        super(VGG11_BN, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def _make_layers(self):
        layers = []
        # Configuration: 'M' is MaxPool, number is channels
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # BatchNorm is added "in-front of activation" (Conv -> BN -> ReLU)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# Helper: Count Params & MACs
# ==========================================
def count_stats(model, input_size=(1, 3, 32, 32)):
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = 0
    
    def hook_fn(module, input, output):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            # MACs = K*K * Cin * Cout * Hout * Wout
            out_h, out_w = output.shape[2], output.shape[3]
            macs = (module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels * out_h * out_w)
            total_macs += macs
        elif isinstance(module, nn.Linear):
            # MACs = Cin * Cout
            macs = module.in_features * module.out_features
            total_macs += macs

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Dummy pass
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
    
    for h in hooks: h.remove()
    return total_params, total_macs

# ==========================================
# Training & Testing
# ==========================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100. * correct / total

def main():
    # Settings
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 1024
    EPOCHS = 50
    LR = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running VGG11 on {DEVICE}")

    # No Data Augmentation (as requested in 5.2)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, download=True, transform=transform),
        batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    model = VGG11_BN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # 1. Calculate and Print Stats
    params, macs = count_stats(model)
    print(f"\n[VGG11 Stats] Params: {params:,} | MACs: {macs:,}\n")

    # 2. Training Loop
    times = []
    accuracies = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        
        # Record cumulative time
        current_time = time.time() - start_time
        times.append(current_time)
        
        # Test accuracy
        acc = test(model, DEVICE, test_loader)
        accuracies.append(acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} | Time: {current_time:.1f}s | Acc: {acc:.2f}%")

    # 3. Save Results
    save_name = "results_vgg11.npz"
    np.savez(save_name, times=np.array(times), accuracies=np.array(accuracies), 
             params=params, macs=macs)
    print(f"\nResults saved to {save_name}")

if __name__ == '__main__':
    main()

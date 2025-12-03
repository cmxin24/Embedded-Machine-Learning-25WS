import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'batch_size': 128,
    'test_batch_size': 1024,
    'epochs': 50,
    'lr': 0.01,
    'dropout_p': 0.0,
    'aug': 'default',
    'seed': 1,
    'log_interval': 200,
    'data_path': '../data',
    'save_model': False,
}

# ==========================================
# Modified Model Definition
# ==========================================
class VGG11(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.0, norm_type='none'):
        """
        norm_type: 'none', 'batch', or 'layer'
        """
        super().__init__()
        self.norm_type = norm_type
        self.layers = self._make_layers(dropout_p)

    def _get_norm_layer(self, channels):
        """Helper to create the correct normalization layer based on config"""
        if self.norm_type == 'batch':
            # 2d implementation for Conv layers
            return nn.BatchNorm2d(channels)
        elif self.norm_type == 'layer':
            # GroupNorm with num_groups=1 is effectively LayerNorm for 2D inputs
            # This satisfies the requirement to use a 2D implementation
            return nn.GroupNorm(num_groups=1, num_channels=channels)
        return None

    def _make_layers(self, dropout_p):
        layers = []
        
        # Helper to build a block: Conv -> [Norm] -> Dropout -> ReLU
        def conv_block(in_c, out_c):
            block = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            
            # Add Normalization if requested
            norm_layer = self._get_norm_layer(out_c)
            if norm_layer:
                block.append(norm_layer)
                
            block.append(nn.Dropout(p=dropout_p))
            block.append(nn.ReLU(inplace=True))
            return block

        # --- Construct VGG Structure ---
        # Block 1
        layers += conv_block(3, 64)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # Block 2
        layers += conv_block(64, 128)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # Block 3
        layers += conv_block(128, 256)
        layers += conv_block(256, 256)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # Block 4
        layers += conv_block(256, 512)
        layers += conv_block(512, 512)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # Block 5
        layers += conv_block(512, 512)
        layers += conv_block(512, 512)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        # Classifier
        layers += [nn.Flatten()]
        
        # Linear 1
        layers += [nn.Linear(512, 4096)]
        # Note: Usually we use BatchNorm1d for Linear, but to keep code simple 
        # and strictly follow "in-front of activation", let's apply ReLU after.
        # For simplicity in this specific VGG task, usually Norm is critical in Conv blocks.
        # However, to be strict, we can add 1D norm here too, or skip for simplicity.
        # Let's stick to standard VGG modifications which focus on the Conv blocks.
        layers += [nn.Dropout(p=dropout_p), nn.ReLU(inplace=True)]
        
        # Linear 2
        layers += [nn.Linear(4096, 4096)]
        layers += [nn.Dropout(p=dropout_p), nn.ReLU(inplace=True)]
        
        # Output
        layers += [nn.Linear(4096, 10)]
        
        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

# ==========================================
# Data Augmentation & Setup (Same as before)
# ==========================================
def get_transforms(aug_type, mean, std):
    # Standard default augmentation
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    
    return running_loss / total, 100. * correct / total

def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return test_loss / total, 100. * correct / total

# ==========================================
# Experiment Runner
# ==========================================
def run_normalization_experiments():
    # Setup Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CONFIG['data_path'], train=True, download=True, 
                         transform=get_transforms('default', mean, std)),
        batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(CONFIG['data_path'], train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std)])),
        batch_size=CONFIG['test_batch_size'], shuffle=False, num_workers=2)

    # Define the 3 experiments
    norm_types = ['none', 'batch', 'layer']
    history = {}

    for norm in norm_types:
        print(f"\n{'='*40}")
        print(f"Starting Experiment: Norm = {norm.upper()}")
        print(f"{'='*40}")
        
        # Initialize Model
        model = VGG11(num_classes=10, dropout_p=CONFIG['dropout_p'], norm_type=norm).to(device)
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9, weight_decay=5e-4)
        
        train_accs = []
        test_accs = []
        
        for epoch in range(1, CONFIG['epochs'] + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test_epoch(model, device, test_loader)
            dt = time.time() - t0
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{CONFIG['epochs']} | Time: {dt:.1f}s | "
                      f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        history[norm] = test_accs

    # Plotting
    plt.figure(figsize=(10, 6))
    for norm, accs in history.items():
        plt.plot(range(1, CONFIG['epochs']+1), accs, label=f'{norm} norm')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison of Normalization Techniques on VGG11')
    plt.legend()
    plt.grid(True)
    plt.savefig('5.1_normalization_comparison.png')
    print("Saved 5.1_normalization_comparison.png")
    plt.show()
    

if __name__ == '__main__':
    run_normalization_experiments()

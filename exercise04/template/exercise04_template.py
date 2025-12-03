"""
Exercise 04: VGG11 on CIFAR-10 with Regularization Techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION - Modify this for experiments
# ==========================================
CONFIG = {
    'batch_size': 128,
    'test_batch_size': 1024,
    'epochs': 30,
    'lr': 0.01,
    'dropout_p': 0.0,        # 4.1: try 0.0, 0.3, 0.5, 0.7
    'L2_reg': None,          # 4.2: try None, 1e-6, 1e-4, 1e-3
    'aug': 'default',        # 4.3: 'none', 'flip', 'crop', 'rotation', 'color', 'default', 'all'
    'seed': 1,
    'log_interval': 200,
    'data_path': '../data',
    'output_dir': './results',
    'save_model': False,
}

# ==========================================
# Model Definition
# ==========================================
class VGG11(nn.Module):
    """VGG11 for CIFAR-10. Dropout is applied before ReLU as requested."""
    
    def __init__(self, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
        layers = []
        def conv(in_c, out_c):
            return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

        layers += [conv(3, 64), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [conv(64, 128), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [conv(128, 256), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(256, 256), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [conv(256, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   conv(512, 512), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Flatten(),
                   nn.Linear(512, 4096), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.Linear(4096, 4096), nn.Dropout(p=dropout_p), nn.ReLU(inplace=True),
                   nn.Linear(4096, 10)]
        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

    def get_last_conv_weights(self):
        """Get weights of last conv layer (for 4.2 histogram)"""
        for layer in reversed(list(self.layers)):
            if isinstance(layer, nn.Conv2d):
                return layer.weight.data.cpu().numpy().flatten()
        return None


# ==========================================
# Data Augmentation (4.3)
# ==========================================
def get_transforms(aug_type, mean, std):
    """Return train transforms based on augmentation type"""
    base = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    
    if aug_type == 'none':
        return transforms.Compose(base)
    elif aug_type == 'flip':
        return transforms.Compose([transforms.RandomHorizontalFlip()] + base)
    elif aug_type == 'crop':
        return transforms.Compose([transforms.RandomCrop(32, padding=4)] + base)
    elif aug_type == 'rotation':
        return transforms.Compose([transforms.RandomRotation(15)] + base)
    elif aug_type == 'color':
        return transforms.Compose([transforms.ColorJitter(0.2, 0.2, 0.2)] + base)
    elif aug_type == 'all':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # default: crop + flip
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


# ==========================================
# Training and Testing Functions
# ==========================================
def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=200):
    """Train for one epoch, return average loss"""
    model.train()
    running_loss = 0.0
    count = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        count += data.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return running_loss / count


def test_epoch(model, device, test_loader):
    """Evaluate model, return (loss, accuracy)"""
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

    test_loss /= total
    accuracy = 100.0 * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


# ==========================================
# Main Training Function
# ==========================================
def run_training(config):
    """
    Run training with given config, return results dict
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(config['seed'])
    
    # Data loaders
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    
    train_transform = get_transforms(config['aug'], cifar10_mean, cifar10_std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    train_kwargs = {'batch_size': config['batch_size'], 'shuffle': True}
    test_kwargs = {'batch_size': config['test_batch_size'], 'shuffle': False}
    if device.type == 'cuda':
        train_kwargs.update({'num_workers': 4, 'pin_memory': True})
        test_kwargs.update({'num_workers': 4, 'pin_memory': True})

    train_dataset = datasets.CIFAR10(config['data_path'], train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(config['data_path'], train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Model and optimizer
    model = VGG11(num_classes=10, dropout_p=config['dropout_p']).to(device)
    weight_decay = float(config['L2_reg']) if config['L2_reg'] is not None else 0.0
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=weight_decay)

    # Results storage
    results = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'epoch_time': [],
        'config': config.copy()
    }

    print(f"Config: dropout={config['dropout_p']}, L2={weight_decay}, aug={config['aug']}")
    print(f"Training for {config['epochs']} epochs...\n")

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch, config['log_interval'])
        test_loss, test_acc = test_epoch(model, device, test_loader)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        results['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")

    # Save model if requested (needed for 4.2 weight histogram)
    results['model'] = model
    
    if config['save_model']:
        os.makedirs(config['output_dir'], exist_ok=True)
        model_path = os.path.join(config['output_dir'], 
                                  f"model_dropout{config['dropout_p']}_L2{weight_decay}_aug{config['aug']}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    return results


# ==========================================
# Plotting Functions
# ==========================================
def plot_single_training(results, title="Training Results"):
    """Plot train/test loss and accuracy for single run"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(results['train_loss']) + 1)
    
    axes[0].plot(epochs, results['train_loss'], 'b-o', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    axes[1].plot(epochs, results['test_loss'], 'r-o', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss')
    axes[1].grid(True)
    
    axes[2].plot(epochs, results['test_accuracy'], 'g-o', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('Test Accuracy')
    axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(results_list, labels, metric='test_accuracy', title='Comparison'):
    """Plot comparison of multiple runs"""
    plt.figure(figsize=(10, 6))
    
    for results, label in zip(results_list, labels):
        epochs = range(1, len(results[metric]) + 1)
        plt.plot(epochs, results[metric], '-o', markersize=3, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weight_histograms(results_list, labels):
    """Plot weight histograms for 4.2 L2 comparison"""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, results, label in zip(axes, results_list, labels):
        weights = results['model'].get_last_conv_weights()
        ax.hist(weights, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label}\nstd={weights.std():.4f}')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle('Last Conv Layer Weight Distribution')
    plt.tight_layout()
    plt.show()


def plot_epoch_times(results_list, labels):
    """Plot average epoch times for 4.3 augmentation comparison"""
    avg_times = [np.mean(r['epoch_time']) for r in results_list]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, avg_times, color='steelblue', edgecolor='black')
    plt.xlabel('Configuration')
    plt.ylabel('Avg Epoch Time (sec)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ==========================================
# Experiment Runner Functions
# ==========================================
def run_dropout_experiments(dropout_values=[0.0, 0.3, 0.5, 0.7], epochs=30):
    """Run 4.1 dropout experiments"""
    results_list = []
    labels = []
    
    for dp in dropout_values:
        print(f"\n{'='*50}")
        print(f"Running dropout = {dp}")
        print('='*50)
        
        config = CONFIG.copy()
        config['dropout_p'] = dp
        config['epochs'] = epochs
        
        results = run_training(config)
        results_list.append(results)
        labels.append(f'dropout={dp}')
    
    # Plot results
    plot_comparison(results_list, labels, 'test_accuracy', 'Dropout: Test Accuracy')
    plot_comparison(results_list, labels, 'test_loss', 'Dropout: Test Loss')
    
    return results_list, labels


def run_l2_experiments(l2_values=[0.0, 1e-6, 1e-4, 1e-3], epochs=30):
    """Run 4.2 L2 regularization experiments"""
    results_list = []
    labels = []
    
    for l2 in l2_values:
        print(f"\n{'='*50}")
        print(f"Running L2 = {l2}")
        print('='*50)
        
        config = CONFIG.copy()
        config['L2_reg'] = l2 if l2 > 0 else None
        config['epochs'] = epochs
        config['save_model'] = True
        
        results = run_training(config)
        results_list.append(results)
        labels.append(f'L2={l2}')
    
    # Plot results
    plot_comparison(results_list, labels, 'test_accuracy', 'L2 Regularization: Test Accuracy')
    plot_comparison(results_list, labels, 'test_loss', 'L2 Regularization: Test Loss')
    plot_weight_histograms(results_list, labels)
    
    return results_list, labels


def run_augmentation_experiments(aug_types=['none', 'flip', 'crop', 'rotation', 'all'], epochs=30):
    """Run 4.3 data augmentation experiments"""
    results_list = []
    labels = []
    
    for aug in aug_types:
        print(f"\n{'='*50}")
        print(f"Running augmentation = {aug}")
        print('='*50)
        
        config = CONFIG.copy()
        config['aug'] = aug
        config['epochs'] = epochs
        
        results = run_training(config)
        results_list.append(results)
        labels.append(f'aug={aug}')
    
    # Plot results
    plot_comparison(results_list, labels, 'test_accuracy', 'Data Augmentation: Test Accuracy')
    plot_comparison(results_list, labels, 'train_loss', 'Data Augmentation: Train Loss')
    plot_epoch_times(results_list, labels)
    
    return results_list, labels


# ==========================================
# Example Usage (Run in Terminal or Jupyter Notebook Cells)
# ==========================================
if __name__ == '__main__':
    # Single training run with current CONFIG
    # results = run_training(CONFIG)
    # plot_single_training(results)
    
    # Or run full experiments:
    results_dropout, labels_dropout = run_dropout_experiments()
    results_l2, labels_l2 = run_l2_experiments()
    results_aug, labels_aug = run_augmentation_experiments()
    pass

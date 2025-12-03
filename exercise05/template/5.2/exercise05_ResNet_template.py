from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from typing import Any, Callable, List, Optional, Type, Union

# ==========================================
# ResNet Basic Block (Fixed Inplace Error)
# ==========================================
class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        
        out = self.relu(out)

        return out

# ==========================================
# ResNet Model
# ==========================================
class ResNet(nn.Module):
    def __init__(self, norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity):
        super().__init__()
        self._norm_layer = norm_layer
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.bn1 = norm_layer(32) 
        
        self.relu = nn.ReLU(inplace=False)

        # Blocks
        self.block1_1 = BasicBlock(32, 32, 1, self._norm_layer)
        self.block1_2 = BasicBlock(32, 32, 1, self._norm_layer)
        self.block1_3 = BasicBlock(32, 32, 1, self._norm_layer)
        
        self.block2_1 = BasicBlock(32, 64, 2, self._norm_layer)
        self.block2_2 = BasicBlock(64, 64, 1, self._norm_layer)
        self.block2_3 = BasicBlock(64, 64, 1, self._norm_layer)
        
        self.block3_1 = BasicBlock(64, 128, 2, self._norm_layer)
        self.block3_2 = BasicBlock(128, 128, 1, self._norm_layer)
        self.block3_3 = BasicBlock(128, 128, 1, self._norm_layer)
        
        # Classifier
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        # Initial Conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Blocks
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        
        x = self.relu(x)
        
        # Pooling (Fixed in previous step)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

# ==========================================
# Training / Testing Functions
# ==========================================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.time(),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()/data.shape[0] ))

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        time.time(),
        epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ResNet Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--L2_reg', type=float, default=1e-4,
                        help='L2_reg (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Standard CIFAR10 Normalization
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)
    
    common_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    dataset_train = datasets.CIFAR10(root='../data', train=True,
                                     download=True, transform=common_transform)
    dataset_test = datasets.CIFAR10(root='../data', train=False, download=True,
                                    transform=common_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    norm_layer = nn.BatchNorm2d
    model = ResNet(norm_layer=norm_layer).to(device)

    if args.L2_reg is not None:
        L2_reg = args.L2_reg
    else:
        L2_reg = 0.
        
    print(f"Hyperparameters: Epochs={args.epochs}, LR={args.lr}, L2={L2_reg}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=L2_reg)

    print(f'Starting training at: {time.time():.4f}')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

if __name__ == '__main__':
    main()
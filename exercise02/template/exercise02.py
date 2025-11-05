from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Linear():
    def __init__(self, in_features: int, out_features: int, batch_size: int, lr=0.1):
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.weight = torch.randn(in_features, out_features) * np.sqrt(1. / in_features)
        self.bias = torch.randn(out_features) * np.sqrt(1. / in_features)
        self.grad_weight = torch.zeros(in_features, out_features)
        self.grad_bias = torch.zeros(out_features)
        self.input = torch.zeros(batch_size, in_features)

    def forward(self, input):
        self.input = input
        output = self.input @ self.weight + self.bias
        return output

    def backward(self, grad_output):
        grad_input = grad_output @ self.weight.T
        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = torch.sum(grad_output, dim=0)
        return grad_input

    def update(self):
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias

class Sigmoid():
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = torch.zeros(batch_size)
        self.output = torch.zeros(batch_size)

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + torch.exp(-self.input))
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (self.output * (1 - self.output))
        return grad_input

def Softmax(input):
    exps = torch.exp(input - torch.max(input, dim=1, keepdim=True)[0])
    output = exps / torch.sum(exps, dim=1, keepdim=True)
    return output

def compute_loss(target, prediction):
    epsilon = 1e-12
    prediction = torch.clamp(prediction, epsilon, 1. - epsilon)
    return -torch.sum(target * torch.log(prediction))/prediction.shape[0]

def compute_gradient(target, prediction):
    return (prediction - target)

class MLP():
    def __init__(self, batch_size, lr):
        super(MLP, self).__init__()
        self.linear0 = Linear(28*28, 512, batch_size, lr)
        self.sigmoid0 = Sigmoid(512, batch_size)
        self.linear1 = Linear(512, 128, batch_size, lr)
        self.sigmoid1 = Sigmoid(128, batch_size)
        self.linear2 = Linear(128, 10, batch_size, lr)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear0.forward(x)
        x = self.sigmoid0.forward(x)
        x = self.linear1.forward(x)
        x = self.sigmoid1.forward(x)
        x = self.linear2.forward(x)
        x = Softmax(x)
        return x

    def backward(self, x):
        x = self.linear2.backward(x)
        x = self.sigmoid1.backward(x)
        x = self.linear1.backward(x)
        x = self.sigmoid0.backward(x)
        x = self.linear0.backward(x)

    def update(self):
        self.linear0.update()
        self.linear1.update()
        self.linear2.update()
    
    def set_lr(self, lr):
        self.linear0.lr = lr
        self.linear1.lr = lr
        self.linear2.lr = lr

def train(args, model, train_loader, epoch):
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] != model.linear0.batch_size:
            model.linear0.batch_size = data.shape[0]
            model.sigmoid0.batch_size = data.shape[0]
            model.linear1.batch_size = data.shape[0]
            model.sigmoid1.batch_size = data.shape[0]
            model.linear2.batch_size = data.shape[0]

        output = model.forward(data)
        loss = compute_loss(target, output)
        train_loss += loss.item()
        gradient = compute_gradient(target, output)
        model.backward(gradient)
        model.update()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / data.shape[0]))
    
    model.linear0.batch_size = args.batch_size
    model.sigmoid0.batch_size = args.batch_size
    model.linear1.batch_size = args.batch_size
    model.sigmoid1.batch_size = args.batch_size
    model.linear2.batch_size = args.batch_size

    return train_loss / len(train_loader)

def test(args, model, test_loader, epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if data.shape[0] != model.linear0.batch_size:
            model.linear0.batch_size = data.shape[0]
            model.sigmoid0.batch_size = data.shape[0]
            model.linear1.batch_size = data.shape[0]
            model.sigmoid1.batch_size = data.shape[0]
            model.linear2.batch_size = data.shape[0]

        output = model.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
       
        target_one_hot = F.one_hot(target, num_classes=10)
        loss = compute_loss(target_one_hot, output)
        test_loss += loss.item()

    model.linear0.batch_size = args.batch_size
    model.sigmoid0.batch_size = args.batch_size
    model.linear1.batch_size = args.batch_size
    model.sigmoid1.batch_size = args.batch_size
    model.linear2.batch_size = args.batch_size

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    return test_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform,
                       target_transform=torchvision.transforms.Compose([
                                 lambda x:torch.LongTensor([x]),
                                 lambda x:F.one_hot(x, 10),
                                 lambda x:x.squeeze()]))

    dataset_test = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=args.batch_size)

    # Part 1: Train with default learning rate
    print("--- Training with default learning rate ---")
    with torch.no_grad():
        model = MLP(args.batch_size, args.lr)
        train_losses = []
        test_losses = []
        test_accuracies = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, train_loader, epoch)
            test_loss, accuracy = test(args, model, test_loader, epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(accuracy)

    # Part 2: Train with varying learning rates
    print("\n--- Training with varying learning rates ---")
    learning_rates = [1.0, 0.5, 0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001]
    lr_results = {}

    for lr in learning_rates:
        print(f"\n--- Training with learning rate: {lr} ---")
        torch.manual_seed(args.seed)
        model = MLP(args.batch_size, lr)
        
        lr_test_accuracies = []
        with torch.no_grad():
            for epoch in range(1, args.epochs + 1):
                train(args, model, train_loader, epoch)
                _, accuracy = test(args, model, test_loader, epoch)
                lr_test_accuracies.append(accuracy)
        lr_results[str(lr)] = lr_test_accuracies
    
    # Save results
    np.savez('results.npz', 
             train_losses=np.array(train_losses), 
             test_losses=np.array(test_losses), 
             test_accuracies=np.array(test_accuracies),
             lr_results=lr_results,
             learning_rates=np.array(learning_rates),
             epochs=args.epochs)

    print("\nTraining complete. Results saved to results.npz")


if __name__ == '__main__':
    main()

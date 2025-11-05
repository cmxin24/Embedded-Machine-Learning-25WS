import numpy as np
import matplotlib.pyplot as plt

def plot_results():
    try:
        data = np.load('results.npz', allow_pickle=True)
    except FileNotFoundError:
        print("Error: results.npz not found. Please run the training script first.")
        return

    train_losses = data['train_losses']
    test_losses = data['test_losses']
    test_accuracies = data['test_accuracies']
    lr_results = data['lr_results'].item() # .item() to get the dictionary back
    learning_rates = data['learning_rates']
    epochs = data['epochs']

    epochs_range = range(1, epochs + 1)

    # Plot 1 & 2: Default training
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs (Default LR)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs. Epochs (Default LR)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('default_training_plots.png')
    print("Saved default training plot to default_training_plots.png")

    # Plot 3: Learning rate comparison
    plt.figure(figsize=(10, 7))
    for lr in learning_rates:
        accuracies = lr_results.get(str(lr))
        if accuracies is not None:
            plt.plot(epochs_range, accuracies, label=f'LR = {lr}')

    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Epochs for Different Learning Rates')
    plt.legend()
    plt.savefig('lr_comparison_plot.png')
    print("Saved learning rate comparison plot to lr_comparison_plot.png")

if __name__ == '__main__':
    plot_results()

import numpy as np
import matplotlib.pyplot as plt
import os


# plot MLP MNIST cpu vs gpu accuracy
cpu = np.load("results_MLP_MNIST_cpu.npz")
gpu = np.load("results_MLP_MNIST_gpu.npz")

plt.plot(cpu["times"], cpu["accuracies"], label="CPU", marker='o')
plt.plot(gpu["times"], gpu["accuracies"], label="GPU", marker='o')

plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("MNIST MLP accuracy over Time")
plt.legend()
plt.grid(True)

output_dir = os.path.join("..", "plot")
output_path = os.path.join(output_dir, "MLP_accuracy_vs_time_MNIST.png")
plt.savefig(output_path)
print(f"saved {output_path}")

# plot MLP vs CNN on accuracy over Epochs
mlp_cifar = np.load("results_MLP_CIFAR10_gpu.npz")
cnn_cifar = np.load("results_CNN_CIFAR10_gpu.npz")

mlp_accu = mlp_cifar["accuracies"]
cnn_accu = cnn_cifar["accuracies"]

epo_mlp = range(1, len(mlp_accu) + 1)
epo_cnn = range(1, len(cnn_accu) + 1)

plt.figure()
plt.plot(epo_mlp, mlp_accu, marker='o', label="MLP CIFAR-10")
plt.plot(epo_cnn, cnn_accu, marker='o', label="CNN CIFAR-10")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP vs CNN: accuracy over epochs on CIFAR-10")
plt.legend()
plt.grid(True)

output_path = os.path.join(output_dir, "MLP_CNN_accuracy_vs_epoch_CIFAR10.png")
plt.savefig(output_path)
print(f"saved {output_path}")

# plot MLP vs CNN on accuracy over time
mlp_time = mlp_cifar["times"]
cnn_time = cnn_cifar["times"]

plt.figure()
plt.plot(mlp_time, mlp_accu, marker='o', label="MLP CIFAR-10")
plt.plot(cnn_time, cnn_accu, marker='o', label="CNN CIFAR-10")

plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.title("MLP vs CNN: ccuracy over time on CIFAR-10")
plt.legend()
plt.grid(True)

output_path = os.path.join(output_dir, "MLP_vs_CNN_accuracy_vs_time_CIFAR10.png")
plt.savefig(output_path)
print(f"saved {output_path}")

# compare different optimizer over epochs
files = {
    "SGD": "results_CNN_CIFAR10_gpu.npz",
    "Adam": "results_Adam_CNN_CIFAR10_gpu.npz",
    "RMSprop": "results_RMSprop_CNN_CIFAR10_gpu.npz"
}
results = {}
for name, path in files.items():
    if os.path.exists(path):
        data = np.load(path)
        results[name] = {
            "times": data["times"],
            "accuracies": data["accuracies"]
        }
plt.figure(figsize=(8, 5))
for name, vals in results.items():
    epochs = range(1, len(vals["accuracies"]) + 1)
    plt.plot(epochs, vals["accuracies"], marker="o", label=name)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Test Accuracy vs Epochs for Different Optimizers with lr0.001")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_dir = os.path.join("..", "plot")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "CIFAR10_CNN_optimizers_accuracy_vs_epoch.png")
plt.savefig(output_path)
print(f"saved {output_path}")

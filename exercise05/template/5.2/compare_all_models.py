import numpy as np
import matplotlib.pyplot as plt
import re
import os

def load_resnet_from_log(logfile):
    """Parses the ResNet log file."""
    accuracies = []
    times = []
    start_time = None
    
    if not os.path.exists(logfile):
        print(f"Warning: {logfile} not found. Skipping ResNet.")
        return [], []

    with open(logfile, "r") as f:
        for line in f:
            # Parse start time
            m_start = re.search(r"Starting training at:\s*([0-9.]+)", line)
            if m_start:
                start_time = float(m_start.group(1))
            
            # Parse Test Accuracy and Epoch Time
            m_test = re.search(
                r"Current time:\s*([0-9.]+);\s*Test Epoch:\s*(\d+),.*Accuracy:\s*\d+/\d+\s*\(([\d.]+)%\)", 
                line
            )
            if m_test:
                acc = float(m_test.group(3))
                accuracies.append(acc)
                
                if start_time is not None:
                    t = float(m_test.group(1))
                    times.append(t - start_time)
                    
    return np.array(times), np.array(accuracies)

def main():
    # ---------------------------------------------
    # 1. Load Data
    # ---------------------------------------------
    
    # === CNN (Exercise 3) ===
    cnn_path = "results_SGD_CNN_CIFAR10_gpu.npz" 
    if os.path.exists(cnn_path):
        cnn_data = np.load(cnn_path)
        cnn_acc = cnn_data["accuracies"]
        cnn_time = cnn_data["times"]
        print(f"Loaded CNN data: {len(cnn_acc)} epochs.")
    else:
        print(f"Error: {cnn_path} not found.")
        cnn_acc, cnn_time = [], []

    # === VGG11 (Exercise 5 - VGG) ===
    vgg_path = "results_vgg11.npz"
    if os.path.exists(vgg_path):
        vgg_data = np.load(vgg_path)
        vgg_acc = vgg_data["accuracies"]
        vgg_time = vgg_data["times"]
        # VGG Params/MACs are stored in the npz file
        vgg_params = int(vgg_data["params"])
        vgg_macs = int(vgg_data["macs"])
        print(f"Loaded VGG data: {len(vgg_acc)} epochs.")
    else:
        print(f"Error: {vgg_path} not found.")
        vgg_acc, vgg_time = [], []
        vgg_params, vgg_macs = 0, 0

    # === ResNet (Exercise 5 - ResNet) ===
    resnet_log = "slurm_output.log"
    res_time, res_acc = load_resnet_from_log(resnet_log)
    if len(res_acc) > 0:
        print(f"Loaded ResNet data: {len(res_acc)} epochs.")

    # ---------------------------------------------
    # 2. Define Stats (Updated from your calculation output)
    # ---------------------------------------------
    # 这些数值来自你刚才运行 calculation_para_MAC.py 的结果
    cnn_params = 2453962
    cnn_macs = 17367680
    
    res_params = 1081482
    res_macs = 162366720

    # VGG 的数据已经从 npz 文件里读取了 (vgg_params, vgg_macs)
    # 如果没读取到，可以用你算出的值：
    if vgg_params == 0:
        vgg_params = 28149514
        vgg_macs = 171679744

    # ---------------------------------------------
    # 3. Print Summary Table (For Discussion)
    # ---------------------------------------------
    print("\n" + "="*65)
    print(f"{'Model':<10} | {'Params':<12} | {'MACs':<15} | {'Best Acc':<10} | {'Total Time':<10}")
    print("-" * 65)
    
    if len(cnn_acc) > 0:
        print(f"{'CNN':<10} | {cnn_params:<12,} | {cnn_macs:<15,} | {max(cnn_acc):.2f}%    | {cnn_time[-1]:.1f}s")
    if len(vgg_acc) > 0:
        print(f"{'VGG11':<10} | {vgg_params:<12,} | {vgg_macs:<15,} | {max(vgg_acc):.2f}%    | {vgg_time[-1]:.1f}s")
    if len(res_acc) > 0:
        print(f"{'ResNet':<10} | {res_params:<12,} | {res_macs:<15,} | {max(res_acc):.2f}%    | {res_time[-1]:.1f}s")
    print("="*65 + "\n")

    # ---------------------------------------------
    # 4. Plot: Accuracy vs Epoch
    # ---------------------------------------------
    plt.figure(figsize=(10, 5))
    
    if len(cnn_acc) > 0:
        plt.plot(range(1, len(cnn_acc)+1), cnn_acc, 'b-o', label=f'CNN (Ex3)', markersize=3, alpha=0.7)
    if len(vgg_acc) > 0:
        plt.plot(range(1, len(vgg_acc)+1), vgg_acc, 'g-s', label=f'VGG11 (BN)', markersize=3, alpha=0.7)
    if len(res_acc) > 0:
        plt.plot(range(1, len(res_acc)+1), res_acc, 'r-^', label=f'ResNet', markersize=3, alpha=0.7)
        
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison: Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('5.2_comparison_epoch.png', dpi=300)
    print("Saved 5.2_comparison_epoch.png")

    # ---------------------------------------------
    # 5. Plot: Accuracy vs Time
    # ---------------------------------------------
    plt.figure(figsize=(10, 5))
    
    if len(cnn_time) > 0:
        plt.plot(cnn_time, cnn_acc, 'b-o', label=f'CNN (Ex3)', markersize=3, alpha=0.7)
    if len(vgg_time) > 0:
        plt.plot(vgg_time, vgg_acc, 'g-s', label=f'VGG11 (BN)', markersize=3, alpha=0.7)
    if len(res_time) > 0:
        plt.plot(res_time, res_acc, 'r-^', label=f'ResNet', markersize=3, alpha=0.7)
        
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison: Accuracy vs Training Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('5.2_comparison_time.png', dpi=300)
    print("Saved 5.2_comparison_time.png")

if __name__ == '__main__':
    main()

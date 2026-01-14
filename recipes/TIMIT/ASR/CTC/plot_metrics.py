import re
import matplotlib.pyplot as plt

# Log files to parse
log_files = [
    "results/timit_train.3498146.out",
    "results/timit_train.3498917.out",
    "results/timit_train.3499489.out",
]

def parse_log_file(filepath):
    """Parse log file and extract epoch, train loss, validation loss, and validation PER."""
    epochs = []
    train_losses = []
    valid_losses = []
    valid_pers = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Match lines like: "epoch: 1, lr: 1.00e-03 - train loss: 2.30 - valid loss: 1.34, valid PER: 40.99"
                match = re.search(
                    r'epoch:\s*(\d+),.*?train loss:\s*([\d.e+-]+)\s*-\s*valid loss:\s*([\d.e+-]+),\s*valid PER:\s*([\d.]+)',
                    line
                )
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    valid_loss = float(match.group(3))
                    valid_per = float(match.group(4))
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    valid_pers.append(valid_per)
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
    
    return epochs, train_losses, valid_losses, valid_pers


# Parse all log files
all_data = {}
for log_file in log_files:
    epochs, train_losses, valid_losses, valid_pers = parse_log_file(log_file)
    all_data[log_file] = {
        'epochs': epochs,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_pers': valid_pers,
    }

# Create three separate plots
# Plot 1: Train Loss
plt.figure(figsize=(10, 6))
for log_file, data in all_data.items():
    label = log_file.split('/')[-1].replace('.out', '')
    plt.plot(data['epochs'], data['train_losses'], marker='o', label=label, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Train Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('train_loss_plot.png', dpi=300)
print("Saved: train_loss_plot.png")
plt.close()

# Plot 2: Validation Loss
plt.figure(figsize=(10, 6))
for log_file, data in all_data.items():
    label = log_file.split('/')[-1].replace('.out', '')
    plt.plot(data['epochs'], data['valid_losses'], marker='s', label=label, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_loss_plot.png', dpi=300)
print("Saved: validation_loss_plot.png")
plt.close()

# Plot 3: Validation PER
plt.figure(figsize=(10, 6))
for log_file, data in all_data.items():
    label = log_file.split('/')[-1].replace('.out', '')
    plt.plot(data['epochs'], data['valid_pers'], marker='^', label=label, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation PER (%)', fontsize=12)
plt.title('Validation Phoneme Error Rate Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_per_plot.png', dpi=300)
print("Saved: validation_per_plot.png")
plt.close()

print("\nAll plots generated successfully!")

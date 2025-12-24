# Import standard libraries
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Modal app and PyTorch libraries
import modal
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm  # For progress bar during training
from torch.utils.tensorboard import SummaryWriter  # For logging

# Import the CNN model
from model import AudioCNN

# Define Modal app
app = modal.App("audio-cnn")

# Create custom Docker image with necessary packages and data
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")  # Install Python dependencies
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])  # Install system dependencies
    .run_commands([  # Download and extract ESC-50 dataset into /opt directory
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")  # Add the local model script
)

# Define volumes to store training data and trained model
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


# Custom dataset class for ESC-50 audio data
class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # Train = folds 1-4, Test = fold 5
        self.metadata = self.metadata[self.metadata['fold'] != 5] if split == 'train' else self.metadata[self.metadata['fold'] == 5]

        # Generate label mappings
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        # Load audio waveform using soundfile (avoids torchcodec dependency)
        waveform, sample_rate = load_audio(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply transform (e.g., spectrogram + masking)
        spectrogram = self.transform(waveform) if self.transform else waveform

        return spectrogram, row['label']


# Mixup augmentation: blend two samples and their labels
def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)  # Random weight for mixup
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Loss function for mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training function to be run remotely on Modal with GPU
# Helper function to load audio using soundfile (avoids torchcodec dependency)
def load_audio(path):
    import soundfile as sf
    data, sr = sf.read(str(path))
    # Convert to torch tensor and add channel dimension if mono
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.T  # soundfile returns (samples, channels), need (channels, samples)
    return waveform, sr


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    from datetime import datetime

    # Logging directory for TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    esc50_dir = Path("/opt/esc50-data")

    # Define training transforms: Spectrogram + masking for augmentation
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    # Validation transform: Spectrogram only (no augmentation)
    val_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, f_min=0, f_max=11025),
        T.AmplitudeToDB()
    )

    # Create datasets and dataloaders
    train_dataset = ESC50Dataset(esc50_dir, esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)
    val_dataset = ESC50Dataset(esc50_dir, esc50_dir / "meta" / "esc50.csv", split="test", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model and training utilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # OneCycleLR scheduler for smooth learning rate changes
    scheduler = OneCycleLR(
        optimizer, max_lr=0.002, epochs=100, steps_per_epoch=len(train_dataloader), pct_start=0.1
    )

    best_accuracy = 0.0

    # Training loop
    print("Starting training")
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/100')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # Apply mixup with 70% probability
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%' )


# Local entrypoint for triggering training from terminal
@app.local_entrypoint()
def main():
    train.remote()  # Trigger training remotely on Modal
    train.remote()  # Run it twice (possibly for experimentation)

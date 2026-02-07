import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import EEGDenoiseAutoencoder, save_model
from signal_utils import generate_eeg_signal, add_noise
import os

class EEGDataset(Dataset):
    """Dataset for EEG signal denoising."""
    
    def __init__(self, num_samples=1000, duration=5.0, sampling_rate=256, noise_types=['gaussian', 'artifacts', 'powerline', 'mixed']):
        self.num_samples = num_samples
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.noise_types = noise_types
        
        # Pre-generate dataset
        self.clean_signals = []
        self.noisy_signals = []
        
        print(f"Generating {num_samples} training samples...")
        for i in range(num_samples):
            # Generate clean signal
            eeg_data = generate_eeg_signal(duration, sampling_rate, num_channels=1)
            clean = eeg_data['signal'][0]  # Single channel
            
            # Add random noise
            noise_type = np.random.choice(noise_types)
            noise_level = np.random.uniform(0.3, 0.8)
            noisy = add_noise(clean, noise_type, noise_level)
            
            self.clean_signals.append(clean)
            self.noisy_signals.append(noisy)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        clean = torch.FloatTensor(self.clean_signals[idx]).unsqueeze(0)  # Add channel dim
        noisy = torch.FloatTensor(self.noisy_signals[idx]).unsqueeze(0)
        return noisy, clean


def train_model(num_epochs=50, batch_size=32, learning_rate=0.001, num_samples=1000):
    """
    Train the EEG denoising autoencoder.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_samples: Number of training samples to generate
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create dataset and dataloader
    dataset = EEGDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = EEGDenoiseAutoencoder()
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, 'models/eeg_denoiser.pth')
            if (epoch + 1) % 5 == 0:
                print(f"  → New best model saved (loss: {best_loss:.6f})")
    
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    print("Model saved to models/eeg_denoiser.pth")
    
    return model


if __name__ == '__main__':
    # Train the model
    print("=" * 60)
    print("EEG Signal Denoising - Model Training")
    print("=" * 60)
    
    trained_model = train_model(
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        num_samples=1000
    )
    
    print("\nTraining complete!")

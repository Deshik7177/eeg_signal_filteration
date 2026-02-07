import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGDenoiseAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for EEG signal denoising.
    
    Architecture:
    - Encoder: 3 conv layers with batch norm and ReLU
    - Decoder: 3 transposed conv layers with batch norm and ReLU
    - Input/Output: (batch, 1, signal_length)
    """
    
    def __init__(self, signal_length=1280):
        super(EEGDenoiseAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            # Layer 1: (1, 1280) -> (16, 640)
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Layer 2: (16, 640) -> (32, 320)
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Layer 3: (32, 320) -> (64, 160)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            # Layer 1: (64, 160) -> (32, 320)
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2: (32, 320) -> (16, 640)
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Layer 3: (16, 640) -> (1, 1280)
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, signal_length)
        
        Returns:
            Denoised signal of same shape as input
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent representation."""
        return self.decoder(z)


def load_model(model_path='models/eeg_denoiser.pth', device='cpu'):
    """
    Load trained model from file.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    model = EEGDenoiseAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_model(model, model_path='models/eeg_denoiser.pth'):
    """
    Save model weights to file.
    
    Args:
        model: PyTorch model to save
        model_path: Path to save model weights
    """
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def denoise_signal(model, noisy_signal, device='cpu'):
    """
    Denoise a signal using trained model.
    
    Args:
        model: Trained autoencoder model
        noisy_signal: Noisy signal array (numpy)
        device: Device to run inference on
    
    Returns:
        Denoised signal as numpy array
    """
    import numpy as np
    
    # Convert to tensor and add batch/channel dimensions
    if len(noisy_signal.shape) == 1:
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0)
    else:
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(1)
    
    noisy_tensor = noisy_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    
    # Convert back to numpy
    denoised = denoised_tensor.squeeze().cpu().numpy()
    
    return denoised

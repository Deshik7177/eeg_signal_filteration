import numpy as np
from scipy import signal as scipy_signal

def generate_eeg_signal(duration=5.0, sampling_rate=256, num_channels=1):
    """
    Generate synthetic EEG signal with multiple frequency components.
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz
        num_channels: Number of EEG channels
    
    Returns:
        dict with 'signal', 'time', 'sampling_rate'
    """
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    
    # Initialize signal
    eeg_signal = np.zeros((num_channels, num_samples))
    
    for ch in range(num_channels):
        # Delta waves (0.5-4 Hz)
        delta = 0.3 * np.sin(2 * np.pi * 2 * time + np.random.rand() * 2 * np.pi)
        
        # Theta waves (4-8 Hz)
        theta = 0.4 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
        
        # Alpha waves (8-13 Hz) - dominant in relaxed state
        alpha = 0.8 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
        
        # Beta waves (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
        
        # Gamma waves (30-100 Hz)
        gamma = 0.1 * np.sin(2 * np.pi * 40 * time + np.random.rand() * 2 * np.pi)
        
        # Combine all components
        eeg_signal[ch] = delta + theta + alpha + beta + gamma
        
        # Add some random variation
        eeg_signal[ch] += 0.05 * np.random.randn(num_samples)
    
    return {
        'signal': eeg_signal,
        'time': time,
        'sampling_rate': sampling_rate
    }

def add_noise(clean_signal, noise_type='gaussian', noise_level=0.5):
    """
    Add different types of noise to EEG signal.
    
    Args:
        clean_signal: Clean EEG signal array
        noise_type: Type of noise ('gaussian', 'artifacts', 'powerline', 'mixed')
        noise_level: Noise intensity (0.0 to 1.0)
    
    Returns:
        Noisy signal array
    """
    signal_shape = clean_signal.shape
    noisy_signal = clean_signal.copy()
    
    if noise_type == 'gaussian':
        # White Gaussian noise
        noise = noise_level * np.random.randn(*signal_shape)
        noisy_signal += noise
        
    elif noise_type == 'artifacts':
        # Simulate eye blink and muscle artifacts
        num_samples = signal_shape[-1]
        
        # Eye blink artifacts (random spikes)
        num_blinks = int(num_samples / 500)
        for _ in range(num_blinks):
            blink_pos = np.random.randint(0, num_samples - 50)
            blink_width = np.random.randint(20, 50)
            blink_amplitude = noise_level * 3.0 * np.random.randn()
            
            # Create blink shape
            blink = blink_amplitude * np.exp(-np.linspace(-2, 2, blink_width)**2)
            
            if len(signal_shape) == 1:
                noisy_signal[blink_pos:blink_pos+blink_width] += blink
            else:
                for ch in range(signal_shape[0]):
                    noisy_signal[ch, blink_pos:blink_pos+blink_width] += blink
        
        # Muscle artifacts (high frequency bursts)
        muscle_noise = noise_level * 0.5 * np.random.randn(*signal_shape)
        # High-pass filter to simulate muscle activity
        if len(signal_shape) == 1:
            b, a = scipy_signal.butter(4, 50, btype='high', fs=256)
            muscle_noise = scipy_signal.filtfilt(b, a, muscle_noise)
        noisy_signal += muscle_noise
        
    elif noise_type == 'powerline':
        # 50/60 Hz powerline interference
        num_samples = signal_shape[-1]
        time = np.linspace(0, num_samples/256, num_samples)
        
        powerline_50hz = noise_level * 0.8 * np.sin(2 * np.pi * 50 * time)
        powerline_60hz = noise_level * 0.5 * np.sin(2 * np.pi * 60 * time)
        
        if len(signal_shape) == 1:
            noisy_signal += powerline_50hz + powerline_60hz
        else:
            for ch in range(signal_shape[0]):
                noisy_signal[ch] += powerline_50hz + powerline_60hz
                
    elif noise_type == 'mixed':
        # Combination of all noise types
        noisy_signal = add_noise(noisy_signal, 'gaussian', noise_level * 0.3)
        noisy_signal = add_noise(noisy_signal, 'artifacts', noise_level * 0.4)
        noisy_signal = add_noise(noisy_signal, 'powerline', noise_level * 0.3)
    
    return noisy_signal

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    Args:
        clean_signal: Clean signal array
        noisy_signal: Noisy signal array
    
    Returns:
        SNR in decibels
    """
    noise = noisy_signal - clean_signal
    
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def bandpass_filter(signal_data, lowcut=0.5, highcut=50, sampling_rate=256, order=4):
    """
    Apply bandpass filter to signal (traditional filtering method for comparison).
    
    Args:
        signal_data: Input signal
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        sampling_rate: Sampling rate
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    
    if len(signal_data.shape) == 1:
        filtered = scipy_signal.filtfilt(b, a, signal_data)
    else:
        filtered = np.zeros_like(signal_data)
        for ch in range(signal_data.shape[0]):
            filtered[ch] = scipy_signal.filtfilt(b, a, signal_data[ch])
    
    return filtered

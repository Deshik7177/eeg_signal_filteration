from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import torch

from signal_utils import generate_eeg_signal, add_noise, calculate_snr, bandpass_filter
from model import load_model, denoise_signal

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model():
    """Load the trained model on startup."""
    global model
    model_path = 'models/eeg_denoiser.pth'
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, device)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model will need to be trained first.")
    else:
        print(f"Model file not found at {model_path}")
        print("Please train the model first by running: python train_model.py")

@app.route('/api/generate-signal', methods=['POST'])
def generate_signal():
    """Generate a synthetic EEG signal."""
    try:
        data = request.get_json(silent=True) or {}
        duration = data.get('duration', 5.0)
        sampling_rate = data.get('sampling_rate', 256)
        
        # Generate signal
        eeg_data = generate_eeg_signal(duration, sampling_rate, num_channels=1)
        
        response = {
            'signal': eeg_data['signal'][0].tolist(),
            'time': eeg_data['time'].tolist(),
            'sampling_rate': eeg_data['sampling_rate'],
            'duration': duration
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-noise', methods=['POST'])
def add_noise_to_signal():
    """Add noise to a clean signal."""
    try:
        data = request.json
        clean_signal = np.array(data['signal'])
        noise_type = data.get('noise_type', 'gaussian')
        noise_level = data.get('noise_level', 0.5)
        
        # Add noise
        noisy_signal = add_noise(clean_signal, noise_type, noise_level)
        
        # Calculate SNR
        snr = calculate_snr(clean_signal, noisy_signal)
        
        response = {
            'noisy_signal': noisy_signal.tolist(),
            'snr': float(snr),
            'noise_type': noise_type,
            'noise_level': noise_level
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter-signal', methods=['POST'])
def filter_signal():
    """Filter a noisy signal using the DL model."""
    try:
        data = request.get_json(silent=True) or {}
        noisy_signal = np.array(data['noisy_signal'])
        clean_signal = np.array(data.get('clean_signal', []))
        
        # Apply DL filtering if model is loaded, else fallback to bandpass filter
        if model is not None:
            filtered_signal = denoise_signal(model, noisy_signal, device)
        else:
            filtered_signal = bandpass_filter(noisy_signal, lowcut=0.5, highcut=40, sampling_rate=256, order=4)
        
        response = {
            'filtered_signal': filtered_signal.tolist(),
        }
        
        # Calculate SNR if clean signal is provided
        if len(clean_signal) > 0:
            # Handle mismatched lengths (e.g., sliding window vs latest chunk)
            clean_len = len(clean_signal)
            if clean_len <= len(noisy_signal):
                # Align to the most recent samples
                noisy_slice = noisy_signal[-clean_len:]
                filtered_slice = filtered_signal[-clean_len:]
                
                snr_before = calculate_snr(clean_signal, noisy_slice)
                snr_after = calculate_snr(clean_signal, filtered_slice)
                snr_improvement = snr_after - snr_before
                
                response['snr_before'] = float(snr_before)
                response['snr_after'] = float(snr_after)
                response['snr_improvement'] = float(snr_improvement)
            else:
                # If clean signal is longer (unlikely), skip metrics for this cycle
                pass
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the model."""
    try:
        model_exists = model is not None
        model_path = 'models/eeg_denoiser.pth'
        model_file_exists = os.path.exists(model_path)
        
        info = {
            'model_loaded': model_exists,
            'model_file_exists': model_file_exists,
            'model_path': model_path,
            'device': str(device),
            'architecture': '1D Convolutional Autoencoder',
            'input_shape': '(batch, 1, 1280)',
            'output_shape': '(batch, 1, 1280)'
        }
        
        if model_exists:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    print("=" * 60)
    print("EEG Signal Filtering - Flask Backend")
    print("=" * 60)
    
    # Initialize model
    initialize_model()
    
    # Run Flask app
    print("\nStarting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

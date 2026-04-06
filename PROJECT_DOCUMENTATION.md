# EEG Signal Filtering Using Deep Learning

## Project Overview
This project demonstrates real-time EEG signal denoising using a 1D Convolutional Autoencoder. It features a Flask backend for signal processing and a modern web frontend for visualization and metric display.

---

## File Structure & Descriptions

- **app.py**: Flask backend exposing REST API endpoints for signal generation, noise addition, filtering (DL or bandpass), and metric calculation. Serves the frontend.
- **model.py**: Defines the 1D Convolutional Autoencoder (EEGDenoiseAutoencoder) and model load/save utilities.
- **signal_utils.py**: Signal processing utilities: EEG generation, noise addition, SNR/PSNR/SSIM/MSE/Efficiency calculations, and bandpass filtering.
- **train_model.py**: Script to train the autoencoder on synthetic EEG data. Produces the model weights file.
- **index.html**: Main frontend UI. Displays live/filtered EEG, metrics, and controls.
- **static/script.js**: Frontend logic for fetching signals/metrics, updating plots/UI, and user controls.
- **static/style.css**: Stylesheet for the frontend.
- **requirements.txt**: Python dependencies.
- **models/eeg_denoiser.pth**: Trained model weights (after running train_model.py).

---

## System Workflow

1. **Signal Generation**: Synthetic EEG signals with multiple frequency bands are generated.
2. **Noise Addition**: Various noise types (Gaussian, artifacts, powerline, mixed) are added.
3. **Denoising**: Noisy signals are filtered using a deep learning autoencoder or a bandpass filter.
4. **Metrics Calculation**: SNR, PSNR, SSIM, MSE, and Efficiency are computed for noisy and filtered signals.
5. **Frontend Visualization**: UI displays live/filtered signals, real-time metrics, and user controls.

---

## Metrics Explained
- **SNR (Signal-to-Noise Ratio)**: Measures signal quality in dB.
- **PSNR (Peak SNR)**: Measures peak error in dB.
- **SSIM (Structural Similarity Index)**: Measures similarity between signals (0-1).
- **MSE (Mean Squared Error)**: Average squared difference between signals.
- **Efficiency**: 1 - (MSE / variance of clean signal), higher is better.

---

## How to Run

1. Install Python 3.8+ and dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model (if not already trained):
   ```
   python train_model.py
   ```
3. Start the backend:
   ```
   python app.py
   ```
4. Open `index.html` in your browser.

---

## For Your Report
- Include this file structure and workflow.
- Add screenshots of the UI.
- Explain each metric and its importance.
- Describe the model architecture and training process.
- Summarize results and possible improvements.

---

**Contact:** SURESH KOTA

---

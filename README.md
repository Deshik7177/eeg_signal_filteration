# EEG Signal Filtering UI

This project provides a live EEG monitoring UI with two synchronized plots and a vitals panel. The top plot shows the raw/noisy signal in real time, and the bottom plot shows a delayed filtered signal (DL model output when available).

## UI Map

Top Status Bar
- ID badge: demo subject identifier.
- Title: application name.
- LAB DEMO: environment label.
- 256 Hz: sampling rate of incoming signal.
- NOTCH: powerline notch reference.
- LIVE REC: live acquisition indicator.
- Clock: system time.

Left Panel (Waveform)
- LIVE SIGNAL (RAW + NOISE): real-time noisy EEG (minimal latency).
- Latency: round-trip update time for the live stream.
- FILTERED SIGNAL (DL / SMOOTHED): denoised output with an intentional delay.
- Latency: round-trip update time for the filtered pipeline.

Right Panel (Vitals)
- SIGNAL QUAL: signal-to-noise ratio (SNR) in dB. Higher is cleaner.
- NOISE RATIO: noise-to-signal ratio (N/S), derived from SNR.
- DOM BAND: dominant EEG band from mock spectrum.
- STATE: heuristic label based on SNR and band.
- BAND/PWR table: per-band power values (mocked).

Controls
- EVENT MARK: drops a vertical marker line on both plots.
- TEST SIGNAL: injects a noise burst to simulate artifacts.

## Notes
- The filtered plot uses the DL model when available; otherwise it falls back to a smoothing filter so the UI always renders.
- The filtered plot is intentionally delayed to mimic real processing latency.
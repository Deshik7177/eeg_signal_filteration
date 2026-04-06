# Deep learning based EEG signal denoising for Brain Computer Interface Applications

## Project Review Explanation

### 1. Introduction
'Deep learning based EEG signal denoising for Brain Computer Interface Applications.' It uses deep learning to clean up brain signals, making them more useful for brain-computer interfaces (BCI).

### 2. 
EEG signals are often very noisy, which makes it hard for computers to understand what the brain is doing. For BCI applications, clean signals are important so the system can work accurately.

### 3. What is EEG and BCI?
EEG is a way to record brain activity using sensors on the head. A BCI is a system that lets people control computers or devices using their brain signals.

### 4. Project Goals
The goal of project is to show how deep learning can remove noise from EEG signals, making them better for BCI systems.

### 5. How the Project Works
- First, the project creates fake EEG signals with different brain wave patterns.
- Then, it adds different types of noise to these signals, like muscle movement or powerline noise.
- Next, it uses two methods to clean the signals: a traditional filter and a deep learning model called an autoencoder.
- Finally, it shows the results live in a web browser, so you can see how well the cleaning works.

### 6. Why is this useful?
This project helps show that deep learning can make BCI systems more accurate by providing cleaner brain signals.


---

## What is EEG?
EEG (Electroencephalography) is a way to record the brain’s electrical activity using sensors on the head. In Brain Computer Interface (BCI) applications, EEG helps us understand and use brain signals to control computers or devices.

## Why Do We Need to Clean EEG Signals for BCI?
EEG signals are very small and can easily get mixed up with unwanted noise, like:
- Blinks or muscle movements (show up as spikes)
- Powerline noise (from electricity)
- Random background noise

For BCI applications, it’s important to have clean EEG signals so the computer can correctly understand what the brain is doing. If the EEG is noisy, the BCI system might make mistakes.

## How Does This Project Clean EEG for BCI?
We use two main methods:
- **Bandpass filter:** Removes noise outside the normal brain wave range.
- **Deep learning autoencoder:** A smart computer model that learns to turn noisy signals back into clean ones, making it easier for BCI systems to use the data.

## What Does This Project Do?
- Makes fake EEG signals with different brain wave patterns
- Adds different types of noise to them
- Cleans the signals using both a filter and a deep learning model
- Shows the results live in a web browser, with simple quality scores

## Why Is This Useful for BCI?
This project shows how deep learning can help clean up brain signals, making BCI systems more accurate and reliable. Clean EEG signals mean better control and communication between the brain and computers or devices.

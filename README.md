# AI/ML Detection of Sibilant Rhonchi (Wheeze) Sounds

This repository implements a complete deep learning + signal processing pipeline to **detect sibilant rhonchi** from stethoscope audio recordings.

## Pipeline Steps
1. Audio input from stethoscope
2. Preprocessing (band-pass filter + wavelet denoise)
3. Nabla coefficient & Legendre–Fenchel feature derivation
4. Empirical Wavelet Transform (EWT)
5. Classification using Gradient Boosting & LSTM

## Run Training
```bash
python main.py
```
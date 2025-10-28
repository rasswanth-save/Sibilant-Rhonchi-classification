import numpy as np, pywt, librosa, scipy.signal as sig

def load_audio(path, sr=8000):
    x, _ = librosa.load(path, sr=sr, mono=True)
    return x

def bandpass(x, fs=8000, low=50, high=2000, order=4):
    sos = sig.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x)

def denoise_wavelet(x):
    coeffs = pywt.wavedec(x, 'db6', level=4)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, 'db6')

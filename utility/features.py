import numpy as np, librosa

def nabla(x):
    return np.concatenate([[0], np.diff(x)])

def compute_frames(x, fs=8000, frame_ms=25, hop_ms=10):
    frame_len = int(frame_ms*fs/1000)
    hop = int(hop_ms*fs/1000)
    return librosa.util.frame(x, frame_length=frame_len, hop_length=hop)

def frame_features(frames, fs=8000):
    feats = []
    for i in range(frames.shape[1]):
        f = frames[:, i]
        rms = np.sqrt(np.mean(f**2))
        zcr = ((f[:-1]*f[1:])<0).sum()/len(f)
        centroid = np.sum(np.fft.rfftfreq(len(f),1/fs)*np.abs(np.fft.rfft(f)))/(np.sum(np.abs(np.fft.rfft(f)))+1e-12)
        entropy = -np.sum(np.abs(np.fft.rfft(f)) * np.log(np.abs(np.fft.rfft(f))+1e-12))
        grad = np.sqrt(np.mean(nabla(f)**2))
        feats.append([rms,zcr,centroid,entropy,grad])
    return np.array(feats)

def aggregate(feats):
    agg = {}
    for i in range(feats.shape[1]):
        col = feats[:,i]
        agg[f'col{i}_mean'] = np.mean(col)
        agg[f'col{i}_std'] = np.std(col)
    return agg

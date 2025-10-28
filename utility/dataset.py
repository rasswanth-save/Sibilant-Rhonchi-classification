import os, glob, numpy as np
from utils import preprocess, features

MAX_SEQ_LEN = 400

def gather_files(data_dir, class_map):
    items = []
    for cname, lab in class_map.items():
        folder = os.path.join(data_dir, cname)
        for f in glob.glob(os.path.join(folder, '*.wav')):
            items.append((f, lab))
    return items

def process_file(path):
    x = preprocess.load_audio(path)
    x = preprocess.bandpass(x)
    x = preprocess.denoise_wavelet(x)
    frames = features.compute_frames(x)
    feats = features.frame_features(frames)
    agg = features.aggregate(feats)
    if feats.shape[0] < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - feats.shape[0], feats.shape[1]))
        feats = np.vstack([feats, pad])
    else:
        feats = feats[:MAX_SEQ_LEN,:]
    return agg, feats

def build_dataset(items):
    agg_list, seq_list, labels = [], [], []
    for path, lab in items:
        agg, seq = process_file(path)
        agg_list.append(agg)
        seq_list.append(seq)
        labels.append(lab)
    keys = sorted(agg_list[0].keys())
    X_agg = np.array([[a[k] for k in keys] for a in agg_list])
    X_seq = np.array(seq_list)
    return X_agg, X_seq, np.array(labels), keys

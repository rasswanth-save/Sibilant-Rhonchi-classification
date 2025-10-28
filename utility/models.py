from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models

def train_gbdt(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sm = SMOTE()
    Xr, yr = sm.fit_resample(Xs, y)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
    clf.fit(Xr, yr)
    return clf, scaler

def build_lstm(seq_len, n_feats, n_classes):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_feats)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

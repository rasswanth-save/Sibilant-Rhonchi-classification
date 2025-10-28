import os, random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils import dataset, preprocess, features, models
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CLASS_MAP = {"normal":0, "heart":1, "lung":2, "sibilant":3}
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("🔍 Gathering dataset...")
    items = dataset.gather_files(DATA_DIR, CLASS_MAP)
    print(f"Found {len(items)} samples.")

    train_items, test_items = train_test_split(items, test_size=0.2,
                                               stratify=[lab for _, lab in items],
                                               random_state=RANDOM_SEED)
    print(" Extracting features")
    X_train_agg, X_train_seq, y_train, keys = dataset.build_dataset(train_items)
    X_test_agg, X_test_seq, y_test, _ = dataset.build_dataset(test_items)

    print("\n Training Gradient Boosting")
    gbdt, scaler = models.train_gbdt(X_train_agg, y_train)
    y_pred = gbdt.predict(scaler.transform(X_test_agg))
    print("\n GBDT Results:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_MAP.keys(), yticklabels=CLASS_MAP.keys())
    plt.title("GBDT Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_DIR, "gbdt_confusion.png"))
    plt.close()

    print("\n Training LSTM...")
    lstm = models.build_lstm(X_train_seq.shape[1], X_train_seq.shape[2], len(CLASS_MAP))
    lstm.fit(X_train_seq, y_train, validation_split=0.1, epochs=10, batch_size=8, verbose=2)
    y_pred_lstm = np.argmax(lstm.predict(X_test_seq), axis=1)
    print("\n LSTM Results:\n", classification_report(y_test, y_pred_lstm))

    lstm.save(os.path.join(RESULTS_DIR, "lstm_model.h5"))
    print(" Models saved in results/")

if __name__ == "__main__":
    main()

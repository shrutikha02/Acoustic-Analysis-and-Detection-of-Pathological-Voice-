# evaluate.py

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_data, split_features_labels, preprocess_data


DATA_PATH = "../data/dataset.csv"
MODEL_PATH = "../models/svm_model.pkl"


def evaluate_model():
    df = load_data(DATA_PATH)
    X, y = split_features_labels(df, target_column="label")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    model, scaler = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model()

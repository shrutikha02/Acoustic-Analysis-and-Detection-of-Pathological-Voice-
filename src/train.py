# train.py

import joblib
from sklearn.svm import SVC
from preprocess import load_data, split_features_labels, preprocess_data


DATA_PATH = "../data/dataset.csv"
MODEL_PATH = "../models/svm_model.pkl"


def train_model():
    df = load_data(DATA_PATH)
    X, y = split_features_labels(df, target_column="label")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    model = SVC(kernel="sigmoid", probability=True)
    model.fit(X_train, y_train)

    joblib.dump((model, scaler), MODEL_PATH)
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()

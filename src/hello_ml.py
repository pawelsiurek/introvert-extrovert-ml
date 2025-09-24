# src/hello_ml.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load a toy dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple model
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("✅ Environment works! Accuracy:", acc)

if __name__ == "__main__":
    main()

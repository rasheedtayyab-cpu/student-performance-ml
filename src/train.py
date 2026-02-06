# src/train.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import load_data
from preprocess import preprocess
from visualize import plot_feature_importances

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def main():
    # Load data
    X, y, feature_names, target_names = load_data()

    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess(X, y)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation set
    preds = model.predict(X_val)

    # Compute accuracy
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")

    # Save predictions
    results = pd.DataFrame({
        "True Label": y_val,
        "Predicted": preds
    })
    results.to_csv("outputs/results.csv", index=False)

    # Save accuracy
    with open("outputs/accuracy.txt", "w") as f:
        f.write(f"Validation Accuracy: {acc:.4f}")

    # Plot and save feature importances
    importances = model.feature_importances_
    plot_feature_importances(importances, feature_names)

if __name__ == "__main__":
    main()

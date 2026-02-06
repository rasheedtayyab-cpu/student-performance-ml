from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import load_data
from preprocess import preprocess

def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess(X, y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

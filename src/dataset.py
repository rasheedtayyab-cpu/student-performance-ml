from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names

# src/visualize.py

import matplotlib.pyplot as plt

def plot_feature_importances(importances, feature_names, save_path="outputs/feature_importances.png"):
    """
    Plot and save feature importances from a trained model.
    Args:
        importances (array): Feature importance scores
        feature_names (list): Names of features
        save_path (str): File path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), feature_names, rotation=90)
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

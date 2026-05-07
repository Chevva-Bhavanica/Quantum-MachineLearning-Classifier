from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from classical_model import get_classical_models
import pandas as pd
from sklearn.metrics import roc_curve, auc
from qiskit.visualization import circuit_drawer
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from quantum_model import build_quantum_classifier
from classical_model import get_classical_models
import matplotlib.pyplot as plt
import numpy as np
import time
# LOAD DATASET
print("\nLoading Iris Dataset...\n")
iris = load_iris()
X = iris.data
y = iris.target
# Binary Classification
# Remove class 2 for simplicity
X = X[y != 2]
y = y[y != 2]
print("Dataset Loaded Successfully!")
print(f"Features Shape: {X.shape}")
print(f"Labels Shape: {y.shape}")
# =========================================================
# DATA PREPROCESSING
# =========================================================

print("\nPreprocessing Data...\n")

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Reduce dimensions to 2
# Quantum circuits work better with fewer features

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X_scaled)

print("Dimensionality Reduced to 2 Features")


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")


# =========================================================
# QUANTUM MODEL
# =========================================================

print("\n======================================")
print("TRAINING QUANTUM CLASSIFIER")
print("======================================\n")

quantum_model = build_quantum_classifier()
# =========================================================
# QUANTUM CIRCUIT VISUALIZATION
# =========================================================

print("\nDisplaying Quantum Feature Map Circuit...\n")

feature_map = quantum_model.feature_map

feature_map.decompose().draw('mpl')

plt.title("Quantum Feature Map")

plt.show()

start_time = time.time()

quantum_model.fit(X_train, y_train)

quantum_training_time = time.time() - start_time

# Predictions
q_predictions = quantum_model.predict(X_test)

# Accuracy
q_accuracy = accuracy_score(y_test, q_predictions)

print(f"Quantum Accuracy: {q_accuracy * 100:.2f}%")

print(f"Quantum Training Time: {quantum_training_time:.2f} seconds")


# =========================================================
# QUANTUM CONFUSION MATRIX
# =========================================================

print("\nQuantum Confusion Matrix:\n")

q_cm = confusion_matrix(y_test, q_predictions)

print(q_cm)

print("\nQuantum Classification Report:\n")

print(classification_report(y_test, q_predictions))


# =========================================================
# CLASSICAL MODELS
# =========================================================

print("\n======================================")
print("TRAINING CLASSICAL MODELS")
print("======================================\n")

models = get_classical_models()

classical_accuracies = {}

training_times = {}

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    classical_accuracies[name] = accuracy * 100

    training_times[name] = training_time

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    print(f"{name} Training Time: {training_time:.4f} seconds")

    print(f"\n{name} Confusion Matrix:\n")

    cm = confusion_matrix(y_test, predictions)

    print(cm)

    print(f"\n{name} Classification Report:\n")

    print(classification_report(y_test, predictions))


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================\n")

print(f"Quantum VQC Accuracy: {q_accuracy * 100:.2f}%")

for name, accuracy in classical_accuracies.items():

    print(f"{name} Accuracy: {accuracy:.2f}%")


# =========================================================
# ACCURACY GRAPH
# =========================================================

model_names = ["Quantum VQC"] + list(classical_accuracies.keys())

accuracies = [q_accuracy * 100] + list(classical_accuracies.values())

plt.figure(figsize=(12, 6))

plt.bar(model_names, accuracies)

plt.ylabel("Accuracy (%)")

plt.xlabel("Models")

plt.title("Quantum vs Classical Machine Learning Models")

plt.xticks(rotation=15)

plt.ylim(0, 110)

for i, value in enumerate(accuracies):

    plt.text(i, value + 1, f"{value:.1f}%", ha='center')

plt.show()


# =========================================================
# TRAINING TIME GRAPH
# =========================================================

time_models = ["Quantum VQC"] + list(training_times.keys())

time_values = [quantum_training_time] + list(training_times.values())

plt.figure(figsize=(12, 6))

plt.bar(time_models, time_values)

plt.ylabel("Training Time (seconds)")

plt.xlabel("Models")

plt.title("Training Time Comparison")

plt.xticks(rotation=15)

for i, value in enumerate(time_values):

    plt.text(i, value + 0.01, f"{value:.2f}s", ha='center')

plt.show()


# =========================================================
# PROJECT COMPLETED
# =========================================================

print("\n======================================")
print("PROJECT EXECUTED SUCCESSFULLY")
print("======================================\n")
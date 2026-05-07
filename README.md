# Quantum Iris Classifier 
A hybrid **Quantum-Classical Machine Learning project** built using Qiskit to classify the Iris dataset and compare performance with classical ML models.

This project demonstrates the basics of **Quantum Machine Learning (QML)** using a Variational Quantum Classifier (VQC) and compares it with traditional machine learning algorithms.

# Project Overview

This project:
- Uses the **Iris dataset** (binary classification for simplicity)
- Builds a **Variational Quantum Classifier (VQC)**
- Combines **quantum circuits + classical optimization**
- Compares results with multiple classical ML models
- Visualizes accuracy, confusion matrix, and training performance

# Technologies Used

- Python 
- :contentReference[oaicite:0]{index=0}
- [Qiskit](https://qiskit.org/?utm_source=chatgpt.com)
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

# Dataset

- Dataset: Iris Dataset (from Scikit-learn)
- Classes used: Binary classification
  - Setosa
  - Versicolor

- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

# Quantum Model

The project uses a:

### :contentReference[oaicite:2]{index=2}

### Key Components:
- Quantum Feature Map (ZZFeatureMap)
- Variational Ansatz (RealAmplitudes)
- Classical Optimizer (COBYLA / SPSA)
- Quantum circuit-based classification

# Project Structure

```bash
quantum-classifier/
│
├── main.py               # Main execution file
├── quantum_model.py      # Quantum VQC model
├── classical_model.py    # Classical ML models
├── requirements.txt      # Dependencies


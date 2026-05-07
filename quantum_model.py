from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA


def build_quantum_classifier():
    # Feature map
    feature_map = ZZFeatureMap(feature_dimension=2, reps=2)

    # Variational circuit (ansatz)
    ansatz = RealAmplitudes(num_qubits=2, reps=2)

    # Quantum sampler
    sampler = StatevectorSampler()

    # Optimizer
    optimizer = COBYLA(maxiter=100)

    # Variational Quantum Classifier
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    return vqc
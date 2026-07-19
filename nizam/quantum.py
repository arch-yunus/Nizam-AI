import numpy as np

HAS_QISKIT = False
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    pass

class QuantumFeatureMap:
    """
    Sovereign Quantum Feature Mapping for Nizam-AI.
    Encodes classical features into quantum states.
    Uses Qiskit if available, otherwise falls back to mathematical unitary rotations.
    """
    def __init__(self, num_qubits: int = 2):
        self.num_qubits = num_qubits
        self.has_qiskit_library = HAS_QISKIT

    def qiskit_encode_and_measure(self, features: np.ndarray) -> np.ndarray:
        """
        Creates a real quantum circuit, encodes parameters, and measures probabilities.
        """
        # Ensure we have at least 2 features, pad if necessary
        feats = np.pad(features, (0, max(0, self.num_qubits - len(features))), 'constant')[:self.num_qubits]
        
        # Scale to [-pi, pi] for angle encoding
        angles = np.clip(feats, -1.0, 1.0) * np.pi
        
        # Create Circuit
        qc = QuantumCircuit(self.num_qubits)
        
        # Superposition
        for q in range(self.num_qubits):
            qc.h(q)
            
        # Angle encoding rotations
        for q in range(self.num_qubits):
            qc.ry(angles[q], q)
            
        # Entanglement (CNOT chain)
        for q in range(self.num_qubits - 1):
            qc.cx(q, q + 1)
            
        # Measure
        qc.measure_all()
        
        # Run Simulator
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=500).result()
        counts = result.get_counts()
        
        # Convert counts to probability vector of dimension 2^num_qubits
        vector_dim = 2 ** self.num_qubits
        probs = np.zeros(vector_dim, dtype=np.float32)
        
        for state_str, count in counts.items():
            # State string like '01'
            idx = int(state_str, 2)
            probs[idx] = count / 500.0
            
        return probs

    def simulated_encode_and_measure(self, features: np.ndarray) -> np.ndarray:
        """
        Pure NumPy simulation of quantum feature mapping rotations.
        Yields stable deterministic fallback probabilities.
        """
        feats = np.pad(features, (0, max(0, self.num_qubits - len(features))), 'constant')[:self.num_qubits]
        angles = np.clip(feats, -1.0, 1.0) * np.pi
        
        # Simulate superposition and rotations
        # For each qubit, compute state vector [cos(theta/2), sin(theta/2)]
        states = []
        for angle in angles:
            # Hadamard state [1/sqrt(2), 1/sqrt(2)] rotated by angle theta/2
            h_state = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
            # Ry rotation matrix: [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]]
            a = angle / 2.0
            rot = np.array([
                [np.cos(a), -np.sin(a)],
                [np.sin(a), np.cos(a)]
            ], dtype=np.float32)
            state = np.dot(rot, h_state)
            states.append(state)
            
        # Tensor product of states
        state_vector = states[0]
        for s in states[1:]:
            state_vector = np.kron(state_vector, s)
            
        # Probabilities are state amplitudes squared
        probs = np.abs(state_vector) ** 2
        # Normalize just in case
        probs = probs / np.sum(probs)
        return probs

    def get_quantum_features(self, features: np.ndarray) -> np.ndarray:
        """
        Exposes consistent API for Nizam Hybrid-AI.
        Returns a probability vector of size 2^num_qubits (e.g. 4 dimensions for 2 qubits).
        """
        features_arr = np.array(features, dtype=np.float32)
        
        if self.has_qiskit_library:
            try:
                return self.qiskit_encode_and_measure(features_arr)
            except Exception:
                # Fallback on any simulator execution failure
                return self.simulated_encode_and_measure(features_arr)
        else:
            return self.simulated_encode_and_measure(features_arr)

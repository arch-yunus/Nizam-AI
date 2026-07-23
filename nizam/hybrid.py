import numpy as np
from nizam.core_wrapper import evaluate_symbolic_clause, quantize_int8, dequantize_int8, quantize_int4, dequantize_int4

class HybridAIModel:
    """
    Hybrid AI Model representing Nizam-AI's core philosophy.
    Combines low-power Symbolic Rules (20W human-like abstract reasoning)
    with Statistical weights (numerical probabilities).
    """
    def __init__(self, num_features, classes, rules=None):
        """
        num_features: dimension of input feature vector
        classes: list of class names/actions (e.g. ['normal', 'anomaly', 'threat'])
        rules: dictionary mapping class index to a list of conditions:
               {class_idx: [(feature_idx, relation, threshold)]}
               relation: 0.0 for <, 1.0 for >
        """
        self.num_features = num_features
        self.classes = classes
        # Initialize statistical weights randomly (simulating pre-trained weights)
        self.weights = np.random.randn(len(classes), num_features).astype(np.float32) * 0.1
        self.biases = np.zeros(len(classes), dtype=np.float32)
        
        # Default rules if none provided
        self.rules = rules if rules is not None else {}

    def statistical_predict(self, features):
        """
        Runs a standard statistical feed-forward evaluation.
        Uses basic dot product and softmax.
        """
        features_arr = np.array(features, dtype=np.float32)
        # Normalize to prevent explosion of logits due to unscaled inputs on Edge devices
        norm = np.linalg.norm(features_arr)
        if norm > 0.0:
            features_arr = features_arr / norm
        logits = np.dot(self.weights, features_arr) + self.biases
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        predicted_class_idx = int(np.argmax(probabilities))
        return predicted_class_idx, probabilities

    def predict(self, features, statistical_weight=0.6, force_symbolic=True):
        """
        Hybrid prediction:
        1. Evaluates symbolic rules. If a rule for a class is fully satisfied, it receives a symbolic boost.
        2. Computes statistical probabilities.
        3. Combines them: combined_score = statistical_weight * prob + (1 - statistical_weight) * symbolic_passed
        4. If force_symbolic is True, any class that explicitly violates its symbolic rules CANNOT be selected.
        
        Returns:
            predicted_class: Name of predicted class/action
            confidence: combined score of the chosen class
            telemetry: Dictionary containing simulated energy/computation measurements
        """
        features_arr = np.array(features, dtype=np.float32)
        stat_idx, stat_probs = self.statistical_predict(features_arr)
        
        symbolic_passes = np.zeros(len(self.classes), dtype=np.float32)
        
        # Evaluate symbolic rules using our Core wrapper
        for class_idx in range(len(self.classes)):
            if class_idx in self.rules and len(self.rules[class_idx]) > 0:
                conditions = self.rules[class_idx]
                passed = evaluate_symbolic_clause(features_arr.tolist(), conditions)
                symbolic_passes[class_idx] = 1.0 if passed else 0.0
            else:
                # No rules defined, default to neutral pass
                symbolic_passes[class_idx] = 0.5

        # Combine predictions
        combined_scores = np.zeros(len(self.classes), dtype=np.float32)
        for i in range(len(self.classes)):
            if force_symbolic and symbolic_passes[i] == 0.0:
                # Rule was active but failed, suppress this class
                combined_scores[i] = -999.0
            else:
                combined_scores[i] = (statistical_weight * stat_probs[i]) + ((1.0 - statistical_weight) * symbolic_passes[i])

        final_class_idx = int(np.argmax(combined_scores))
        confidence = float(combined_scores[final_class_idx])
        
        # Calculate simulated energy consumption (Nizam 20W brain-like vs Kaba 300W model)
        # Low parameters and symbolic routing saves 90%+ energy.
        ops_nizam = len(features) * len(self.classes) + len(self.rules) * 3  # Extremely light
        ops_traditional = len(features) * 4096 * 3 + 4096 * 4096 * 2       # Simulating huge LLM
        
        energy_nizam_joules = ops_nizam * 1e-9 * 20.0     # 20W scaled
        energy_traditional_joules = ops_traditional * 1e-9 * 300.0 # 300W scaled
        
        telemetry = {
            "ops_nizam": ops_nizam,
            "ops_traditional": ops_traditional,
            "energy_nizam_joules": energy_nizam_joules,
            "energy_traditional_joules": energy_traditional_joules,
            "efficiency_gain_ratio": ops_traditional / max(1, ops_nizam),
            "symbolic_passes": symbolic_passes.tolist(),
            "statistical_probs": stat_probs.tolist()
        }
        
        return self.classes[final_class_idx], confidence, telemetry

    def quantize_model(self, scale=0.01, zero_point=0):
        """
        Quantizes weights to 8-bit to simulate running on resource-constrained Edge hardware.
        """
        flat_weights = self.weights.flatten().tolist()
        quantized_flat = quantize_int8(flat_weights, scale, zero_point)
        dequantized_flat = dequantize_int8(quantized_flat.tolist(), scale, zero_point)
        self.weights = dequantized_flat.reshape(self.weights.shape)
        return len(flat_weights)  # Return number of quantized params

    def quantize_model_int4(self, scale=0.01, zero_point=0):
        """
        Quantizes weights to 4-bit to simulate running on resource-constrained Edge hardware.
        """
        flat_weights = self.weights.flatten().tolist()
        quantized_flat = quantize_int4(flat_weights, scale, zero_point)
        dequantized_flat = dequantize_int4(quantized_flat.tolist(), scale, zero_point)
        self.weights = dequantized_flat.reshape(self.weights.shape)
        return len(flat_weights)  # Return number of quantized params

    def predict_quantum(self, features, statistical_weight=0.6, force_symbolic=True):
        """
        Quantum-enhanced Hybrid AI prediction:
        1. Encodes classical features into a quantum state vector representation.
        2. Performs classification on the quantum-state feature map.
        3. Combines with symbolic logic constraints.
        """
        from nizam.quantum import QuantumFeatureMap
        qmap = QuantumFeatureMap(num_qubits=2)
        q_features = qmap.get_quantum_features(features)
        
        # Pad or slice quantum features to match the network input dimension
        if q_features.shape[0] < self.num_features:
            q_features = np.pad(q_features, (0, self.num_features - q_features.shape[0]), 'constant')
        else:
            q_features = q_features[:self.num_features]
            
        pred_class, confidence, telemetry = self.predict(q_features.tolist(), statistical_weight, force_symbolic)
        telemetry["quantum_mapping_applied"] = True
        telemetry["quantum_feature_state"] = q_features.tolist()
        return pred_class, confidence, telemetry


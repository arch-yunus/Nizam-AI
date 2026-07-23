import hashlib
import hmac
import time
import numpy as np
from typing import Dict, List, Tuple, Any

class SovereignNodeSigner:
    """
    Sovereign Cryptographic Signer for Nizam-AI Nodes.
    Simulates Post-Quantum Cryptography (PQC Dilithium/Kyber representation)
    by signing weight updates with secure cryptographic hashes (HMAC-SHA256 & SHA-512).
    """
    def __init__(self, node_id: str, secret_key: str = None):
        self.node_id = node_id
        self.secret_key = secret_key if secret_key else f"NIZAM-PQC-KEY-{node_id}-{hashlib.sha256(node_id.encode()).hexdigest()[:8]}"

    def sign_payload(self, weights: np.ndarray, biases: np.ndarray) -> Dict[str, Any]:
        """
        Calculates SHA-512 fingerprint of weight matrices and signs with HMAC.
        """
        weights_bytes = weights.tobytes()
        biases_bytes = biases.tobytes()
        payload_hash = hashlib.sha512(weights_bytes + biases_bytes).hexdigest()
        
        signature = hmac.new(
            self.secret_key.encode(),
            payload_hash.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "node_id": self.node_id,
            "payload_hash": payload_hash,
            "pqc_signature": f"PQC-DILITHIUM-V1::{signature}",
            "timestamp": time.time()
        }

    def verify_signature(self, weights: np.ndarray, biases: np.ndarray, signature_meta: Dict[str, Any]) -> bool:
        """
        Verifies if the received weight update payload matches the signature.
        """
        weights_bytes = weights.tobytes()
        biases_bytes = biases.tobytes()
        expected_hash = hashlib.sha512(weights_bytes + biases_bytes).hexdigest()
        
        if expected_hash != signature_meta.get("payload_hash"):
            return False

        sig_raw = signature_meta.get("pqc_signature", "").replace("PQC-DILITHIUM-V1::", "")
        expected_sig = hmac.new(
            self.secret_key.encode(),
            expected_hash.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_sig, sig_raw)


class ByzantineRobustFilter:
    """
    Byzantine-Robust Weight Filter for Federated Learning.
    Detects and filters out malicious/poisoned node weight submissions
    using multiple defense modes: MAD (Median Absolute Deviation), TRIMMED_MEAN, or KRUM.
    """
    def __init__(self, filter_type: str = "MAD", mad_threshold: float = 3.5, f: int = 1):
        """
        filter_type: "MAD", "TRIMMED_MEAN", or "KRUM"
        mad_threshold: threshold for MAD/Trimmed Mean Z-score filtering
        f: number of tolerated Byzantine nodes for KRUM
        """
        self.filter_type = filter_type.upper()
        self.mad_threshold = mad_threshold
        self.f = f

    def filter_updates(self, node_ids: List[str], weights_list: List[np.ndarray], biases_list: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Inspects node updates and filters out poisoned/anomalous weights.
        Returns (valid_node_ids, valid_weights, valid_biases, rejected_node_ids)
        """
        n = len(weights_list)
        if n <= 2:
            # Not enough nodes to compute reliable statistics, accept all
            return node_ids, weights_list, biases_list, []

        # Flatten weights and biases for distance calculations
        flat_updates = []
        for w, b in zip(weights_list, biases_list):
            flat_updates.append(np.concatenate([w.flatten(), b.flatten()]))
        
        flat_updates = np.array(flat_updates, dtype=np.float32)

        valid_ids, valid_w, valid_b, rejected_ids = [], [], [], []

        if self.filter_type == "KRUM":
            # KRUM: Choose a single representative update that minimizes the sum of L2 distances
            # to its n - f - 2 nearest neighbors.
            num_neighbors = max(1, n - self.f - 2)
            scores = []
            
            for i in range(n):
                dists = []
                for j in range(n):
                    if i != j:
                        dists.append(float(np.linalg.norm(flat_updates[i] - flat_updates[j])))
                dists.sort()
                # Sum the closest neighbors
                scores.append(sum(dists[:num_neighbors]))
                
            best_idx = int(np.argmin(scores))
            
            for idx in range(n):
                if idx == best_idx:
                    valid_ids.append(node_ids[idx])
                    valid_w.append(weights_list[idx])
                    valid_b.append(biases_list[idx])
                else:
                    rejected_ids.append(node_ids[idx])
                    
            return valid_ids, valid_w, valid_b, rejected_ids

        elif self.filter_type == "TRIMMED_MEAN":
            # TRIMMED_MEAN: Calculate coordinate-wise trimmed mean, then filter outliers
            # Trim 10% from both ends (minimum 1 element)
            trim_pct = 0.1
            k = max(1, int(n * trim_pct))
            
            trimmed_mean = []
            for col in range(flat_updates.shape[1]):
                sorted_vals = np.sort(flat_updates[:, col])
                trimmed_vals = sorted_vals[k:-k] if n > 2 * k else sorted_vals
                trimmed_mean.append(np.mean(trimmed_vals))
                
            trimmed_mean = np.array(trimmed_mean, dtype=np.float32)
            
            # Calculate distance of each node from the trimmed mean
            distances = np.array([float(np.linalg.norm(u - trimmed_mean)) for u in flat_updates], dtype=np.float32)
            med_dist = np.median(distances)
            mad = np.median(np.abs(distances - med_dist))
            
            for idx, dist in enumerate(distances):
                if mad > 1e-8:
                    mod_z_score = 0.6745 * abs(dist - med_dist) / mad
                else:
                    mod_z_score = 0.0 if abs(dist - med_dist) < 1e-5 else 999.0

                if mod_z_score > self.mad_threshold:
                    rejected_ids.append(node_ids[idx])
                else:
                    valid_ids.append(node_ids[idx])
                    valid_w.append(weights_list[idx])
                    valid_b.append(biases_list[idx])
                    
            return valid_ids, valid_w, valid_b, rejected_ids

        else: # Default is "MAD" L2 distance from median
            median_weight = np.median(flat_updates, axis=0)
            distances = np.array([float(np.linalg.norm(u - median_weight)) for u in flat_updates], dtype=np.float32)
            med_dist = np.median(distances)
            mad = np.median(np.abs(distances - med_dist))
            
            for idx, dist in enumerate(distances):
                if mad > 1e-8:
                    mod_z_score = 0.6745 * abs(dist - med_dist) / mad
                else:
                    mod_z_score = 0.0 if abs(dist - med_dist) < 1e-5 else 999.0

                if mod_z_score > self.mad_threshold:
                    rejected_ids.append(node_ids[idx])
                else:
                    valid_ids.append(node_ids[idx])
                    valid_w.append(weights_list[idx])
                    valid_b.append(biases_list[idx])

            return valid_ids, valid_w, valid_b, rejected_ids

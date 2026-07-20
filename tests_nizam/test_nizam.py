import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from nizam.core_wrapper import quantize_int8, dequantize_int8, quantize_binary, evaluate_symbolic_clause
from nizam.hybrid import HybridAIModel
from nizam.edge import EdgeDeviceSimulator
from nizam.distributed import NizamNode, NizamFederatedServer
from nizam.pilots.defense import UAVSwarmTacticalSimulator
from nizam.pilots.health import EdgeDiagnosticsAssistant, SmartDrugDiscoverySimulator
from nizam.pilots.education import PersonalizedLearningAssistant

class TestNizamAI(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_quantization(self):
        # Quantize float inputs to signed 8-bit integers
        float_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        scale = 0.02
        zero_point = 10
        
        quantized = quantize_int8(float_values, scale, zero_point)
        self.assertEqual(len(quantized), 5)
        
        # Test back-and-forth dequantization
        dequantized = dequantize_int8(quantized, scale, zero_point)
        # Check difference is small (due to quantization errors)
        for original, restored in zip(float_values, dequantized):
            self.assertLess(abs(original - restored), scale * 2.0)

    def test_binary_quantization(self):
        float_values = [-1.5, 0.0, 2.5, -0.1]
        binary = quantize_binary(float_values)
        self.assertEqual(binary[0], -1)
        self.assertEqual(binary[1], 1)
        self.assertEqual(binary[2], 1)
        self.assertEqual(binary[3], -1)

    def test_evaluate_symbolic_clause(self):
        features = [15.0, 120.0, 0.0]
        # Rules:
        # feature_idx 0 > 10.0  => (0, 1.0, 10.0)
        # feature_idx 1 < 150.0 => (1, 0.0, 150.0)
        conditions = [
            (0, 1.0, 10.0),
            (1, 0.0, 150.0)
        ]
        result = evaluate_symbolic_clause(features, conditions)
        self.assertTrue(result)

        # Violation test
        conditions_fail = [
            (0, 1.0, 20.0)  # 15.0 is not > 20.0
        ]
        result_fail = evaluate_symbolic_clause(features, conditions_fail)
        self.assertFalse(result_fail)

    def test_hybrid_ai_model(self):
        rules = {
            1: [(0, 1.0, 50.0)]  # class index 1 requires feature 0 > 50
        }
        model = HybridAIModel(
            num_features=2,
            classes=['normal', 'alert'],
            rules=rules
        )
        
        # Violates rule of alert
        pred, conf, telemetry = model.predict([10.0, 1.0], force_symbolic=True)
        self.assertEqual(pred, 'normal')
        
        # Satisfies rule of alert (should trigger alert if statistical weights or combined logic favors it)
        # Let's ensure no crash happens
        pred2, conf2, telemetry2 = model.predict([80.0, 1.0], force_symbolic=True)
        self.assertIn(pred2, ['normal', 'alert'])
        self.assertGreater(telemetry2["efficiency_gain_ratio"], 1.0)

    def test_edge_device_simulator(self):
        sim = EdgeDeviceSimulator(battery_capacity_mah=4000, nominal_voltage=3.7)
        runtime_nizam = sim.calculate_battery_runtime(20.0)
        runtime_server = sim.calculate_battery_runtime(300.0)
        self.assertGreater(runtime_nizam, runtime_server)

        level_after, energy = sim.simulate_discharge(100.0, 3600.0, 20.0)  # 1 hour at 20W
        self.assertLess(level_after, 100.0)

    def test_federated_learning(self):
        # 2 features, 2 classes
        server = NizamFederatedServer(num_features=2, num_classes=2)
        
        # Node 1
        node1_x = np.array([[1.0, 2.0], [3.0, 4.0]])
        node1_y = np.array([0, 1])
        node1 = NizamNode("Node-Ankara", node1_x, node1_y, 2, 2)
        
        # Node 2
        node2_x = np.array([[5.0, 6.0], [7.0, 8.0]])
        node2_y = np.array([1, 0])
        node2 = NizamNode("Node-Istanbul", node2_x, node2_y, 2, 2)
        
        server.register_node(node1)
        server.register_node(node2)
        
        round_info = server.run_federated_round(local_epochs=1, learning_rate=0.01)
        self.assertEqual(round_info["round_status"], "Success")
        self.assertIn("Node-Ankara", round_info["active_nodes"])

    def test_pilots(self):
        # Test Defense
        swarm = UAVSwarmTacticalSimulator(num_drones=3)
        step_res = swarm.step()
        self.assertEqual(len(step_res["drones"]), 3)
        self.assertIn("target", step_res)
        
        # Test Health
        diag = EdgeDiagnosticsAssistant()
        diag_res = diag.diagnose_cell(600.0, 40.0, 0.9, 0.8)
        self.assertEqual(diag_res["diagnosis"], "Malignant")
        
        drug = SmartDrugDiscoverySimulator()
        compounds = drug.screen_compounds(num_candidates=3)
        self.assertEqual(len(compounds), 3)

        # Test Education
        edu = PersonalizedLearningAssistant()
        edu_res = edu.assess_student_state(90.0, 95.0, 95.0, 10.0)
        self.assertEqual(edu_res["recommended_mode"], "ChallengeMode")

    def test_quantum_integration(self):
        from nizam.quantum import QuantumFeatureMap
        qmap = QuantumFeatureMap(num_qubits=2)
        q_feats = qmap.get_quantum_features([0.5, -0.2])
        self.assertEqual(len(q_feats), 4) # 2^2 = 4 state probabilities
        self.assertAlmostEqual(sum(q_feats), 1.0, places=4)

        # Test predict_quantum
        model = HybridAIModel(num_features=4, classes=['class1', 'class2'])
        pred, conf, telemetry = model.predict_quantum([0.5, -0.2])
        self.assertIn(pred, ['class1', 'class2'])
        self.assertTrue(telemetry["quantum_mapping_applied"])
        self.assertEqual(len(telemetry["quantum_feature_state"]), 4)

    def test_storage_and_security(self):
        from nizam.security import SovereignNodeSigner, ByzantineRobustFilter
        from nizam.storage import NizamDatabase
        
        # Test PQC Signer
        signer = SovereignNodeSigner("Node-Test")
        w = np.ones((2, 2), dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        sig = signer.sign_payload(w, b)
        self.assertTrue(signer.verify_signature(w, b, sig))
        
        # Test tampering
        w_tampered = w * 2.0
        self.assertFalse(signer.verify_signature(w_tampered, b, sig))

        # Test Byzantine Filter
        b_filter = ByzantineRobustFilter()
        node_ids = ["Node1", "Node2", "Node3", "NodeMalicious"]
        weights = [np.ones((2,2))*0.1, np.ones((2,2))*0.11, np.ones((2,2))*0.09, np.ones((2,2))*100.0]
        biases = [np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2)*50.0]
        valid_ids, valid_w, valid_b, rejected_ids = b_filter.filter_updates(node_ids, weights, biases)
        self.assertIn("NodeMalicious", rejected_ids)
        self.assertNotIn("NodeMalicious", valid_ids)

        # Test Storage DB
        db = NizamDatabase(":memory:") # In-memory database for testing
        db.log_security_event("TEST_EVENT", "Node-Test", "SUCCESS", "Security test pass.")
        logs = db.get_recent_security_logs(limit=5)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["event_type"], "TEST_EVENT")

    def test_robotics_and_pqc_tunnel(self):
        from nizam.robotics import ExtendedKalmanFilter, RobotKinematicsEngine
        from nizam.pqc_tunnel import PQCHybridTunnel
        
        # Test EKF Filter
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.predict(ax=1.0, ay=0.5)
        ekf.update(gps_x=2.0, gps_y=1.0)
        state = ekf.get_state()
        self.assertIn("x", state)
        self.assertIn("uncertainty_x", state)
        
        # Test Kinematics Engine
        engine = RobotKinematicsEngine()
        step_res = engine.step_simulation(gps_jammed=False)
        self.assertEqual(step_res["step"], 1)
        self.assertIsNotNone(step_res["noisy_gps"])

        # Test PQC Tunnel
        tunnel = PQCHybridTunnel("Node-A", "Node-B")
        payload = {"cmd": "TAKTIK_ROTA", "val": 100}
        pkt = tunnel.encrypt_packet(payload)
        is_authentic, dec_payload = tunnel.decrypt_packet(pkt)
        self.assertTrue(is_authentic)
        self.assertEqual(dec_payload["cmd"], "TAKTIK_ROTA")

if __name__ == '__main__':
    unittest.main()

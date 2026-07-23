import os
import sys
from flask import Flask, render_template, jsonify, request
import numpy as np

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nizam.edge import EdgeDeviceSimulator
from nizam.distributed import NizamNode, NizamFederatedServer
from nizam.pilots.defense import UAVSwarmTacticalSimulator
from nizam.pilots.health import EdgeDiagnosticsAssistant, SmartDrugDiscoverySimulator
from nizam.pilots.education import PersonalizedLearningAssistant
from nizam.integrations import KureClient, T3AIClient, EnSosyalClient
from nizam.storage import NizamDatabase
from nizam.robotics import RobotKinematicsEngine
from nizam.pqc_tunnel import PQCHybridTunnel

app = Flask(__name__)

# Core instances
edge_sim = EdgeDeviceSimulator()
swarm_sim = UAVSwarmTacticalSimulator(num_drones=6)
health_diag = EdgeDiagnosticsAssistant()
drug_discovery = SmartDrugDiscoverySimulator()
learning_assistant = PersonalizedLearningAssistant()
robotics_engine = RobotKinematicsEngine()
db = NizamDatabase()

kure_client = KureClient()
t3ai_client = T3AIClient()
ensosyal_client = EnSosyalClient()

# State variables
robotics_gps_jammed = False
robotics_gps_spoofed = False

# Federated learning setup
fed_server = NizamFederatedServer(num_features=4, num_classes=3)
nodes_data = {
    "Tubitak-Node": (np.random.randn(20, 4), np.random.randint(0, 3, 20)),
    "Aselsan-Node": (np.random.randn(15, 4), np.random.randint(0, 3, 15)),
    "Havelsan-Node": (np.random.randn(25, 4), np.random.randint(0, 3, 25)),
    "Roketsan-Node": (np.random.randn(10, 4), np.random.randint(0, 3, 10))
}

for name, (x, y) in nodes_data.items():
    node = NizamNode(name, x, y, num_features=4, num_classes=3)
    fed_server.register_node(node)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    hw = request.args.get('hardware_target', 'Generic Mobile')
    quant = request.args.get('quantization_mode', 'FP32')
    telemetry = edge_sim.get_system_telemetry(target_hardware=hw, quantization_mode=quant)
    db.log_telemetry(
        cpu=telemetry["cpu_percent"],
        ram=telemetry["real_process_ram_mb"],
        battery_hours=telemetry["runtime_nizam_hours"],
        power_w=telemetry["nizam_edge_simulated_power_w"]
    )
    return jsonify(telemetry)


@app.route('/api/telemetry/profile', methods=['GET'])
def profile_inference():
    hw = request.args.get('hardware_target', 'Generic Mobile')
    quant = request.args.get('quantization_mode', 'FP32')
    params = int(request.args.get('params', 1000000))
    res = edge_sim.profile_inference(num_params=params, target_hardware=hw, quantization_mode=quant)
    return jsonify(res)


@app.route('/api/federated/config', methods=['POST'])
def federated_config():
    data = request.json
    filter_type = data.get('filter_type', 'MAD')
    fed_server.set_byzantine_filter_type(filter_type)
    return jsonify({"status": "success", "filter_type": filter_type.upper()})


@app.route('/api/swarm/step', methods=['GET'])
def swarm_step():
    res = swarm_sim.step()
    return jsonify(res)


@app.route('/api/swarm/toggle-jamming', methods=['POST'])
def toggle_jamming():
    state = swarm_sim.toggle_jamming()
    db.log_security_event("ELECTRONIC_WARFARE_TOGGLE", "Swarm-Simulator", "WARNING" if state else "NORMAL", f"Jamming state toggled to {state}")
    return jsonify({"jamming_active": state})


@app.route('/api/federated/step', methods=['POST'])
def federated_step():
    res = fed_server.run_federated_round(local_epochs=2, learning_rate=0.05)
    
    round_no = request.json.get('round', 1)
    if round_no % 3 == 0:
        ensosyal_client.publish_post(
            title=f"Nizam-AI FL Round {round_no} Başarıldı",
            content=f"Dağıtık sistemlerimizde {len(fed_server.nodes)} kurum düğümünün katkısıyla milli yapay zeka modelimizin güncellenmesi tamamlandı.",
            author_node="Merkez-Sunucu"
        )
        
    return jsonify({
        "status": res["round_status"],
        "nodes": res["active_nodes"],
        "rejected_byzantine_nodes": res.get("rejected_byzantine_nodes", []),
        "samples": res["total_samples_processed"],
        "weights_norm": round(res["global_weights_norm"], 4),
        "privacy": res["privacy_metric"],
        "recent_social_posts": ensosyal_client.posts_history[-3:]
    })


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    area = float(data.get('area', 450))
    perimeter = float(data.get('perimeter', 35))
    texture = float(data.get('texture', 0.5))
    mitotic = float(data.get('mitotic', 0.2))
    
    res = health_diag.diagnose_cell(area, perimeter, texture, mitotic)
    db.log_health_diagnosis(area, perimeter, texture, mitotic, res["diagnosis"], res["confidence"])
    return jsonify(res)


@app.route('/api/screen-drugs', methods=['GET'])
def screen_drugs():
    candidates = drug_discovery.screen_compounds(num_candidates=6)
    return jsonify({"candidates": candidates})


@app.route('/api/education/chat', methods=['POST'])
def edu_chat():
    data = request.json
    attention = float(data.get('attention', 80))
    math_score = float(data.get('math', 75))
    science_score = float(data.get('science', 80))
    fatigue = float(data.get('fatigue', 20))
    
    res = learning_assistant.assess_student_state(attention, math_score, science_score, fatigue)
    return jsonify(res)


@app.route('/api/query-kure', methods=['GET'])
def query_kure():
    keyword = request.args.get('keyword', 'quantization_loss')
    res = kure_client.query_fact(keyword)
    return jsonify(res)


@app.route('/api/t3ai-chat', methods=['POST'])
def t3ai_chat():
    data = request.json
    prompt = data.get('prompt', 'Nizam-AI nedir?')
    res = t3ai_client.generate(prompt)
    return jsonify(res)


@app.route('/api/security/audit', methods=['GET'])
def security_audit():
    logs = db.get_recent_security_logs(limit=15)
    summary = db.get_audit_summary()
    return jsonify({
        "audit_logs": logs,
        "summary": summary
    })


@app.route('/api/robotics/step', methods=['GET'])
def robotics_step():
    global robotics_gps_jammed, robotics_gps_spoofed
    res = robotics_engine.step_simulation(gps_jammed=robotics_gps_jammed, gps_spoofed=robotics_gps_spoofed)
    return jsonify(res)


@app.route('/api/robotics/toggle-gps-jamming', methods=['POST'])
def toggle_robotics_gps():
    global robotics_gps_jammed
    robotics_gps_jammed = not robotics_gps_jammed
    db.log_security_event("ROBOTICS_GPS_JAMMING", "EKF-Nav-Engine", "WARNING" if robotics_gps_jammed else "NORMAL", f"GPS outage state set to {robotics_gps_jammed}")
    return jsonify({"gps_jammed": robotics_gps_jammed})


@app.route('/api/robotics/toggle-gps-spoofing', methods=['POST'])
def toggle_robotics_spoofing():
    global robotics_gps_spoofed
    robotics_gps_spoofed = not robotics_gps_spoofed
    db.log_security_event("ROBOTICS_GPS_SPOOFING", "EKF-Nav-Engine", "WARNING" if robotics_gps_spoofed else "NORMAL", f"GPS spoofing state set to {robotics_gps_spoofed}")
    return jsonify({"gps_spoofed": robotics_gps_spoofed})


@app.route('/api/pqc/test-tunnel', methods=['POST'])
def test_pqc_tunnel():
    tunnel = PQCHybridTunnel("Aselsan-Node", "Roketsan-Node")
    sample_data = {"telemetry": "TACTICAL_DRONE_POS", "coordinates": [40.1, 32.8], "status": "SECURE"}
    
    encrypted_packet = tunnel.encrypt_packet(sample_data)
    is_valid, decrypted_data = tunnel.decrypt_packet(encrypted_packet)
    
    db.log_security_event("PQC_TUNNEL_TEST", "Aselsan->Roketsan", "SUCCESS" if is_valid else "FAILED", f"Decrypted payload matches: {is_valid}")
    
    return jsonify({
        "packet": encrypted_packet,
        "verification_success": is_valid,
        "decrypted_payload": decrypted_data
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

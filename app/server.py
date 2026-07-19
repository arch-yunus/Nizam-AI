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

app = Flask(__name__)

# Core instances
edge_sim = EdgeDeviceSimulator()
swarm_sim = UAVSwarmTacticalSimulator(num_drones=6)
health_diag = EdgeDiagnosticsAssistant()
drug_discovery = SmartDrugDiscoverySimulator()
learning_assistant = PersonalizedLearningAssistant()

kure_client = KureClient()
t3ai_client = T3AIClient()
ensosyal_client = EnSosyalClient()

# Federated learning setup
fed_server = NizamFederatedServer(num_features=4, num_classes=3)
# Register a few simulated nodes representing local institutions
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
    telemetry = edge_sim.get_system_telemetry()
    return jsonify(telemetry)


@app.route('/api/swarm/step', methods=['GET'])
def swarm_step():
    res = swarm_sim.step()
    return jsonify(res)


@app.route('/api/swarm/toggle-jamming', methods=['POST'])
def toggle_jamming():
    state = swarm_sim.toggle_jamming()
    return jsonify({"jamming_active": state})


@app.route('/api/federated/step', methods=['POST'])
def federated_step():
    res = fed_server.run_federated_round(local_epochs=2, learning_rate=0.05)
    
    # Push update to En Sosyal if progress is made
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


if __name__ == '__main__':
    # Run the local Flask server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)

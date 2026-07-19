import random
import math
from typing import List, Dict, Tuple
from nizam.hybrid import HybridAIModel

class UAVDrone:
    """
    Represents an autonomous UAV (Unmanned Aerial Vehicle) operating inside a swarm.
    Decisions are evaluated locally using Edge AI principles.
    """
    def __init__(self, drone_id: int, x: float, y: float):
        self.drone_id = drone_id
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.speed = 2.0
        self.sensor_range = 100.0
        
        # Local hybrid decision model
        # Features: [dist_to_target, flock_distance, battery_level, is_jammed]
        # Classes: ['Cohesion' (move closer), 'Disperse' (spread out), 'ReturnToBase' (charge), 'Evade' (electronic warfare)]
        # Rules:
        # If is_jammed (feature 3) > 0.5 => Evade (class 3)
        # If battery_level (feature 2) < 20.0 => ReturnToBase (class 2)
        rules = {
            3: [(3, 1.0, 0.5)],  # Jammed -> Evade
            2: [(2, 0.0, 20.0)], # Low battery -> ReturnToBase
        }
        self.decision_model = HybridAIModel(
            num_features=4,
            classes=['Cohesion', 'Disperse', 'ReturnToBase', 'Evade'],
            rules=rules
        )
        self.battery = 100.0

    def update_state(self, target_x: float, target_y: float, neighbors: List['UAVDrone'], is_jammed: bool = False):
        """
        Simulates local Edge AI decision making.
        1. Prepares sensory features.
        2. Queries local Hybrid AI model.
        3. Updates velocities/position accordingly.
        """
        # Feature 0: Distance to target
        dist_to_target = math.hypot(target_x - self.x, target_y - self.y)
        
        # Feature 1: Average distance to flock neighbors
        if neighbors:
            avg_neigh_dist = sum(math.hypot(n.x - self.x, n.y - self.y) for n in neighbors) / len(neighbors)
        else:
            avg_neigh_dist = 50.0 # Nominal distance
            
        # Decrease battery slowly
        self.battery = max(0.0, self.battery - 0.05)
        
        # Prepare feature vector
        features = [
            dist_to_target,
            avg_neigh_dist,
            self.battery,
            1.0 if is_jammed else 0.0
        ]
        
        # Local model inference (Hybrid!)
        action, confidence, telemetry = self.decision_model.predict(features, statistical_weight=0.7)
        
        # Update velocities based on decision
        if action == 'Evade':
            # Run away in random direction or opposite of target
            angle = random.uniform(0, 2 * math.pi)
            self.vx = math.cos(angle) * self.speed * 1.5
            self.vy = math.sin(angle) * self.speed * 1.5
        elif action == 'ReturnToBase':
            # Head towards (0, 0) base
            dist_to_base = math.hypot(self.x, self.y)
            if dist_to_base > 1.0:
                self.vx = -self.x / dist_to_base * self.speed
                self.vy = -self.y / dist_to_base * self.speed
            else:
                self.vx, self.vy = 0.0, 0.0
                self.battery = 100.0 # Recharged at base
        elif action == 'Disperse':
            # Move away from neighbors
            if neighbors:
                dx = sum(self.x - n.x for n in neighbors)
                dy = sum(self.y - n.y for n in neighbors)
                norm = math.hypot(dx, dy)
                if norm > 0.1:
                    self.vx = dx / norm * self.speed
                    self.vy = dy / norm * self.speed
        else: # Cohesion / Seek target
            # Move toward target with slight noise
            dx = target_x - self.x
            dy = target_y - self.y
            norm = math.hypot(dx, dy)
            if norm > 0.1:
                # Add tiny flocking separation
                sep_x, sep_y = 0.0, 0.0
                for n in neighbors:
                    d = math.hypot(self.x - n.x, self.y - n.y)
                    if d < 15.0 and d > 0: # Too close!
                        sep_x += (self.x - n.x) / d
                        sep_y += (self.y - n.y) / d
                
                # Combine vectors
                self.vx = (dx / norm * 0.7 + sep_x * 0.3) * self.speed
                self.vy = (dy / norm * 0.7 + sep_y * 0.3) * self.speed

        # Move
        self.x += self.vx
        self.y += self.vy
        
        # Keep inside bounds (0 to 800)
        self.x = max(10.0, min(790.0, self.x))
        self.y = max(10.0, min(590.0, self.y))
        
        return {
            "drone_id": self.drone_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "battery": round(self.battery, 1),
            "action": action,
            "confidence": round(confidence, 2),
            "telemetry": telemetry
        }


class UAVSwarmTacticalSimulator:
    """
    Simulates a flock of UAVs tracking down a target drone.
    Shows collaborative Edge intelligence.
    """
    def __init__(self, num_drones: int = 5):
        self.drones = [UAVDrone(i, random.uniform(100, 700), random.uniform(100, 500)) for i in range(num_drones)]
        self.target_x = 400.0
        self.target_y = 300.0
        self.target_vx = 1.0
        self.target_vy = 0.8
        self.is_jammed_state = False

    def toggle_jamming(self):
        self.is_jammed_state = not self.is_jammed_state
        return self.is_jammed_state

    def step(self) -> Dict:
        """
        Executes one simulator step.
        """
        # Move target randomly
        self.target_x += self.target_vx + random.uniform(-1.5, 1.5)
        self.target_y += self.target_vy + random.uniform(-1.5, 1.5)
        
        # Keep target in bounds
        if self.target_x < 50 or self.target_x > 750:
            self.target_vx *= -1
            self.target_x = max(50.0, min(750.0, self.target_x))
        if self.target_y < 50 or self.target_y > 550:
            self.target_vy *= -1
            self.target_y = max(50.0, min(550.0, self.target_y))

        drone_states = []
        
        for drone in self.drones:
            # Find neighbors within sensor range
            neighbors = [d for d in self.drones if d.drone_id != drone.drone_id and math.hypot(d.x - drone.x, d.y - drone.y) < drone.sensor_range]
            
            # Update individual drone
            state = drone.update_state(self.target_x, self.target_y, neighbors, is_jammed=self.is_jammed_state)
            drone_states.append(state)

        # Average energy computations saved across the swarm
        total_nizam_energy = sum(ds['telemetry']['energy_nizam_joules'] for ds in drone_states)
        total_trad_energy = sum(ds['telemetry']['energy_traditional_joules'] for ds in drone_states)
        
        return {
            "target": {"x": round(self.target_x, 1), "y": round(self.target_y, 1)},
            "drones": drone_states,
            "jammed": self.is_jammed_state,
            "summary_telemetry": {
                "total_nizam_energy_joules": round(total_nizam_energy, 6),
                "total_traditional_energy_joules": round(total_trad_energy, 6),
                "efficiency_gain_ratio": round(total_trad_energy / max(1e-10, total_nizam_energy), 1)
            }
        }

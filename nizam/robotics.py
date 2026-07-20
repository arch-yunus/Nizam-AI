import numpy as np
import math
from typing import Dict, List, Tuple, Any

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) Engine for Nizam-AI Autonomous Robotics & UAV Navigation.
    Provides state estimation (position, velocity) under noisy sensor measurements (GPS/IMU/LiDAR)
    and maintains inertial dead-reckoning navigation when GPS is jammed.
    """
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        
        # State vector [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)
        
        # State Transition Matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Process Noise Covariance Q
        q_var = 0.05
        self.Q = np.eye(4, dtype=np.float64) * q_var
        
        # Measurement Matrix H (GPS measures x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Measurement Noise Covariance R (GPS noise)
        r_var = 2.5
        self.R = np.eye(2, dtype=np.float64) * r_var
        
        # Error Covariance Matrix P
        self.P = np.eye(4, dtype=np.float64) * 10.0

    def predict(self, ax: float = 0.0, ay: float = 0.0):
        """
        Predict step: x_k = F * x_{k-1} + B * u_k
        u_k: acceleration inputs [ax, ay]
        """
        # Control Input Matrix B
        B = np.array([
            [0.5 * (self.dt ** 2), 0.0],
            [0.0, 0.5 * (self.dt ** 2)],
            [self.dt, 0.0],
            [0.0, self.dt]
        ], dtype=np.float64)
        
        u = np.array([[ax], [ay]], dtype=np.float64)
        
        # Predict State
        self.x = np.dot(self.F, self.x) + np.dot(B, u)
        # Predict Covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, gps_x: float, gps_y: float):
        """
        Update step using GPS measurement z = [gps_x, gps_y]^T
        """
        z = np.array([[gps_x], [gps_y]], dtype=np.float64)
        
        # Innovation y = z - H * x
        y = z - np.dot(self.H, self.x)
        
        # Innovation Covariance S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Updated State Estimate
        self.x = self.x + np.dot(K, y)
        
        # Updated Covariance P = (I - K * H) * P
        I = np.eye(4, dtype=np.float64)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self) -> Dict[str, float]:
        return {
            "x": float(self.x[0, 0]),
            "y": float(self.x[1, 0]),
            "vx": float(self.x[2, 0]),
            "vy": float(self.x[3, 0]),
            "uncertainty_x": float(math.sqrt(max(0, self.P[0, 0]))),
            "uncertainty_y": float(math.sqrt(max(0, self.P[1, 1])))
        }


class RobotKinematicsEngine:
    """
    Simulates a physical robot / UAV trajectory under noisy sensors and GPS outages.
    Demonstrates Edge EKF navigation under Electronic Warfare (EW).
    """
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dt=0.1)
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_vx = 5.0
        self.true_vy = 3.0
        self.step_count = 0

    def step_simulation(self, gps_jammed: bool = False) -> Dict[str, Any]:
        self.step_count += 1
        
        # True physics motion
        ax = math.sin(self.step_count * 0.1) * 0.5
        ay = math.cos(self.step_count * 0.1) * 0.5
        
        self.true_vx += ax * 0.1
        self.true_vy += ay * 0.1
        self.true_x += self.true_vx * 0.1
        self.true_y += self.true_vy * 0.1
        
        # Predict with IMU acceleration
        self.ekf.predict(ax, ay)
        
        # Measure
        noisy_gps_x = self.true_x + np.random.normal(0, 1.5)
        noisy_gps_y = self.true_y + np.random.normal(0, 1.5)
        
        if not gps_jammed:
            self.ekf.update(noisy_gps_x, noisy_gps_y)

        state = self.ekf.get_state()
        
        # Calculate positioning error
        raw_gps_error = math.hypot(noisy_gps_x - self.true_x, noisy_gps_y - self.true_y) if not gps_jammed else 999.0
        ekf_error = math.hypot(state["x"] - self.true_x, state["y"] - self.true_y)
        
        return {
            "step": self.step_count,
            "true_position": {"x": round(self.true_x, 2), "y": round(self.true_y, 2)},
            "noisy_gps": {"x": round(noisy_gps_x, 2), "y": round(noisy_gps_y, 2)} if not gps_jammed else None,
            "ekf_estimated": {"x": round(state["x"], 2), "y": round(state["y"], 2)},
            "gps_jammed": gps_jammed,
            "raw_gps_error_m": round(raw_gps_error, 2),
            "ekf_error_m": round(ekf_error, 2),
            "uncertainty_m": round((state["uncertainty_x"] + state["uncertainty_y"]) / 2.0, 2)
        }

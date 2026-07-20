import sqlite3
import os
import time
from typing import Dict, List, Any

class NizamDatabase:
    """
    Local SQLite database engine for Nizam-AI.
    Persists system telemetry, federated learning rounds, health diagnostic logs,
    and security audit events locally on the Edge node.
    """
    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            db_path = os.path.join(base_dir, "nizam_local_storage.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()

    def _get_connection(self):
        return self.conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Telemetry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_percent REAL,
                    ram_mb REAL,
                    battery_hours REAL,
                    power_w REAL
                )
            """)
            
            # Federated Learning History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS federated_rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER,
                    active_nodes TEXT,
                    weights_norm REAL,
                    sample_count INTEGER,
                    byzantine_detected INTEGER,
                    timestamp REAL
                )
            """)
            
            # Health Diagnostics Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_diagnoses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cell_area REAL,
                    cell_perimeter REAL,
                    texture REAL,
                    mitotic REAL,
                    diagnosis TEXT,
                    confidence REAL
                )
            """)
            
            # Security Audit Logs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    node_id TEXT,
                    status TEXT,
                    details TEXT
                )
            """)
            conn.commit()

    def log_telemetry(self, cpu: float, ram: float, battery_hours: float, power_w: float = 20.0):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_logs (timestamp, cpu_percent, ram_mb, battery_hours, power_w)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), cpu, ram, battery_hours, power_w))
            conn.commit()

    def log_federated_round(self, round_no: int, nodes: List[str], weights_norm: float, samples: int, byzantine_detected: bool = False):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO federated_rounds (round_number, active_nodes, weights_norm, sample_count, byzantine_detected, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (round_no, ",".join(nodes), weights_norm, samples, 1 if byzantine_detected else 0, time.time()))
            conn.commit()

    def log_health_diagnosis(self, area: float, perimeter: float, texture: float, mitotic: float, diagnosis: str, confidence: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO health_diagnoses (timestamp, cell_area, cell_perimeter, texture, mitotic, diagnosis, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), area, perimeter, texture, mitotic, diagnosis, confidence))
            conn.commit()

    def log_security_event(self, event_type: str, node_id: str, status: str, details: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO security_audit_logs (timestamp, event_type, node_id, status, details)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), event_type, node_id, status, details))
            conn.commit()

    def get_recent_security_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, node_id, status, details
                FROM security_audit_logs
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
        return [{
            "timestamp": time.strftime("%H:%M:%S", time.localtime(r[0])),
            "event_type": r[1],
            "node_id": r[2],
            "status": r[3],
            "details": r[4]
        } for r in rows]

    def get_audit_summary(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM telemetry_logs")
            telemetry_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM federated_rounds")
            fl_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM health_diagnoses")
            health_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM security_audit_logs")
            sec_count = cursor.fetchone()[0]

        return {
            "db_path": self.db_path,
            "telemetry_records": telemetry_count,
            "federated_rounds": fl_count,
            "health_diagnoses": health_count,
            "security_events": sec_count
        }

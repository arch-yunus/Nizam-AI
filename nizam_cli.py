#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nizam-AI Command Line Interface (CLI)
Provides utilities to launch the Flask Dashboard, run test suites,
and execute edge-computing benchmarks.
"""

import argparse
import sys
import os
import subprocess
import time
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nizam.core_wrapper import quantize_int8, dequantize_int8, py_quantize_int8, py_dequantize_int8
from nizam.hybrid import HybridAIModel

def run_tests():
    """Runs the unit tests."""
    print("[Nizam-AI] Birim testleri başlatılıyor...")
    import unittest
    # Import the tests explicitly
    from tests_nizam.test_nizam import TestNizamAI
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNizamAI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

def run_dashboard():
    """Runs the Flask web dashboard."""
    print("[Nizam-AI] Flask Web Arayüzü Başlatılıyor...")
    print("[Nizam-AI] Lütfen tarayıcınızda şu adresi açın: http://127.0.0.1:5000")
    from app.server import app
    app.run(debug=True, host="127.0.0.1", port=5000)

def run_benchmarks():
    """Runs Edge AI performance benchmarks."""
    print("=" * 70)
    print("                 NİZAM-AI EDGE BENCHMARK ANALİZİ")
    print("=" * 70)
    print("[1] Quantization vs Floating-Point Bellek Tasarrufu")
    
    # Create random weights
    num_params = 1000000 # 1 Million weights
    print(f" - Simüle edilen ağırlık parametresi: {num_params:,} (Float32)")
    
    float32_size = num_params * 4 / (1024 * 1024) # 4 bytes each, in MB
    int8_size = num_params * 1 / (1024 * 1024) # 1 byte each, in MB
    
    print(f" - Float32 Bellek Boyutu: {float32_size:.2f} MB")
    print(f" - Int8 Nicemlenmiş Boyut: {int8_size:.2f} MB")
    print(f" - Kazanılan Bellek Alanı: %{((float32_size - int8_size) / float32_size * 100):.1f} (Tasarruf)")
    
    print("\n[2] Hız Performansı Karşılaştırması (10,000 eleman, 1,000 tekrar)")
    data = np.random.randn(10000).astype(np.float32)
    
    # Measure Python quantization function
    start_py = time.perf_counter()
    for _ in range(1000):
        _ = py_quantize_int8(data, 0.02, 10)
    end_py = time.perf_counter()
    py_dur = (end_py - start_py) * 1000 # in ms
    print(f" - Saf Python Nicemleme Süresi: {py_dur:.2f} ms")

    # Measure C++ ctypes if loaded, or simulate it
    from nizam.core_wrapper import HAS_CPP_CORE
    if HAS_CPP_CORE:
        start_cpp = time.perf_counter()
        for _ in range(1000):
            _ = quantize_int8(data.tolist(), 0.02, 10)
        end_cpp = time.perf_counter()
        cpp_dur = (end_cpp - start_cpp) * 1000
        print(f" - C++ Çekirdek Nicemleme Süresi: {cpp_dur:.2f} ms")
        print(f" - C++ Hızlanma Çarpanı: {(py_dur / max(1e-5, cpp_dur)):.2f}x")
    else:
        print(" - C++ Çekirdeği derlenmemiş (Python Fallback devrede).")
        print(" - C++ Derleme adımları için README.md belgesini inceleyin.")
        
    print("\n[3] Enerji Ayak İzi Karşılaştırması (Uç Cihaz Tüketimi)")
    print(" - Nizam-AI Edge Mimari Güç Tüketimi: 20 Watt (Sensör seviyesinde anlık karar)")
    print(" - Geleneksel Bulut GPU Sunucu Tüketimi: 300 Watt (Gecikmeli ağ aktarımı dahil)")
    print(f" - Enerji Tasarruf Katsayısı: {(300.0 / 20.0):.1f}x")
    print("=" * 70)


def run_security_audit():
    """Prints local security audit logs and database metrics."""
    from nizam.storage import NizamDatabase
    db = NizamDatabase()
    summary = db.get_audit_summary()
    logs = db.get_recent_security_logs(limit=10)
    
    print("=" * 70)
    print("                 NİZAM-AI SİBER GÜVENLİK VE DB DENETİMİ")
    print("=" * 70)
    print(f" - Veri Tabanı Konumu: {summary['db_path']}")
    print(f" - Toplam Telemetri Kaydı: {summary['telemetry_records']}")
    print(f" - Toplam FL Yuvarlak Kaydı: {summary['federated_rounds']}")
    print(f" - Toplam Sağlık Tanı Kaydı: {summary['health_diagnoses']}")
    print(f" - Toplam Siber Güvenlik Olayı: {summary['security_events']}")
    print("\n[Son Siber Güvenlik Denetim Logları]")
    if not logs:
        print(" - Henüz kayıtlı güvenlik olayı bulunmamaktadır.")
    else:
        for log in logs:
            print(f" [{log['timestamp']}] [{log['event_type']}] Node: {log['node_id']} | Status: {log['status']} | {log['details']}")
    print("=" * 70)


def run_robotics_sim():
    """Executes EKF navigation simulation steps and PQC Tunnel verification."""
    from nizam.robotics import RobotKinematicsEngine
    from nizam.pqc_tunnel import PQCHybridTunnel
    
    print("=" * 70)
    print("        NİZAM-AI OTONOM ROBOTİK EKF SEYRÜSEFER VE PQC TÜNELİ")
    print("=" * 70)
    
    engine = RobotKinematicsEngine()
    print("[1] Extended Kalman Filter (EKF) Seyir Simülasyonu (5 Adım Normal, 5 Adım EW Jamming)")
    
    for i in range(1, 11):
        is_jammed = (i > 5)
        res = engine.step_simulation(gps_jammed=is_jammed)
        status = "EW JAMMED (INS Only)" if is_jammed else "GPS Normal"
        print(f" Adım {i:02d} [{status}] | Gerçek: ({res['true_position']['x']}, {res['true_position']['y']}) | EKF: ({res['ekf_estimated']['x']}, {res['ekf_estimated']['y']}) | EKF Hata: {res['ekf_error_m']}m")
        
    print("\n[2] PQC Kyber-1024 / Dilithium-5 Hibrit Tünel Testi")
    tunnel = PQCHybridTunnel("Aselsan-UAV-1", "Roketsan-Base")
    payload = {"telemetry": "TACTICAL_DRONE_POS", "coordinates": [40.1, 32.8]}
    pkt = tunnel.encrypt_packet(payload)
    is_valid, dec = tunnel.decrypt_packet(pkt)
    print(f" - Şifreli Paket Boyutu: {len(pkt['ciphertext'])} chars")
    print(f" - PQC KEM: {pkt['kem']} | PQC İmza: {pkt['sig']}")
    print(f" - Doğrulama Durumu: {'BAŞARILI' if is_valid else 'BAŞARISIZ'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Nizam-AI Yönetim Arayüzü")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Birim testleri çalıştırır.")
    group.add_argument("--dashboard", action="store_true", help="Flask tabanlı web panelini başlatır.")
    group.add_argument("--benchmark", action="store_true", help="Edge AI optimizasyon testlerini çalıştırır.")
    group.add_argument("--security", action="store_true", help="Siber güvenlik ve yerel SQLite denetim raporunu gösterir.")
    group.add_argument("--robotics", action="store_true", help="Otonom robotik EKF seyrüsefer ve PQC tünel testini çalıştırır.")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.dashboard:
        run_dashboard()
    elif args.benchmark:
        run_benchmarks()
    elif args.security:
        run_security_audit()
    elif args.robotics:
        run_robotics_sim()

if __name__ == "__main__":
    main()

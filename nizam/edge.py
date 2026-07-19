import time
import sys
import psutil
import os

class EdgeDeviceSimulator:
    """
    Simulates edge device parameters, telemetry, and power profile.
    Highlights Nizam-AI's low power (20W) consumption and localized processing constraints.
    """
    def __init__(self, battery_capacity_mah=5000, nominal_voltage=3.7):
        """
        battery_capacity_mah: Default smartphone/edge battery (e.g. 5000 mAh)
        nominal_voltage: 3.7V standard
        """
        self.battery_capacity_mah = battery_capacity_mah
        self.voltage = nominal_voltage
        # Total battery energy in Wh = (mAh * V) / 1000
        self.battery_energy_wh = (battery_capacity_mah * nominal_voltage) / 1000.0
        self.battery_level = 100.0  # Percentage

    def calculate_battery_runtime(self, power_consumption_watts):
        """
        Calculates how many hours the device can run under a given power load.
        """
        if power_consumption_watts <= 0:
            return float('inf')
        return self.battery_energy_wh / power_consumption_watts

    def simulate_discharge(self, initial_level, active_seconds, power_consumption_watts):
        """
        Returns new battery percentage after running for active_seconds at specified power.
        """
        energy_consumed_wh = (power_consumption_watts * (active_seconds / 3600.0))
        pct_lost = (energy_consumed_wh / self.battery_energy_wh) * 100.0
        new_level = max(0.0, initial_level - pct_lost)
        return float(new_level), float(energy_consumed_wh)

    def get_system_telemetry(self):
        """
        Retrieves actual OS memory and cpu usage, combined with simulated edge values.
        """
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        ram_mb = ram_bytes / (1024 * 1024)
        
        # Telemetry comparison
        return {
            "real_process_ram_mb": round(ram_mb, 2),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "nizam_edge_simulated_power_w": 20.0,
            "traditional_server_power_w": 300.0,
            "runtime_nizam_hours": round(self.calculate_battery_runtime(20.0), 2),
            "runtime_traditional_hours": round(self.calculate_battery_runtime(300.0), 2)
        }

    @staticmethod
    def calculate_quantization_savings(original_weights_float32):
        """
        Calculates memory compression ratio when converting Float32 to Int8.
        """
        original_size_bytes = len(original_weights_float32) * 4  # 32 bits = 4 bytes
        quantized_size_bytes = len(original_weights_float32) * 1  # 8 bits = 1 byte
        
        return {
            "original_size_kb": round(original_size_bytes / 1024.0, 3),
            "quantized_size_kb": round(quantized_size_bytes / 1024.0, 3),
            "compression_ratio": 4.0,
            "memory_saved_percent": 75.0
        }

import time
import sys
import psutil
import os

class EdgeDeviceSimulator:
    """
    Simulates edge device parameters, telemetry, and power profile.
    Highlights Nizam-AI's low power (20W) consumption and localized processing constraints.
    Supports profiling for different hardware architectures (ARM, RISC-V, Jetson, Pi).
    """
    HARDWARE_PROFILES = {
        "ARM Cortex-M4/M7": {
            "power_w": 0.15,
            "base_latency_ms": 15.0,
            "fp32_cycle_multiplier": 8.0,
            "int8_cycle_multiplier": 1.5,
            "int4_cycle_multiplier": 0.5,
            "memory_overhead_kb": 128.0
        },
        "RISC-V Vector": {
            "power_w": 0.12,
            "base_latency_ms": 12.0,
            "fp32_cycle_multiplier": 6.0,
            "int8_cycle_multiplier": 1.0,
            "int4_cycle_multiplier": 0.3,
            "memory_overhead_kb": 96.0
        },
        "Raspberry Pi 5": {
            "power_w": 4.5,
            "base_latency_ms": 2.2,
            "fp32_cycle_multiplier": 1.0,
            "int8_cycle_multiplier": 0.25,
            "int4_cycle_multiplier": 0.10,
            "memory_overhead_kb": 2048.0
        },
        "NVIDIA Jetson Orin Nano": {
            "power_w": 12.0,
            "base_latency_ms": 0.8,
            "fp32_cycle_multiplier": 0.5,
            "int8_cycle_multiplier": 0.08,
            "int4_cycle_multiplier": 0.03,
            "memory_overhead_kb": 8192.0
        },
        "Generic Mobile": {
            "power_w": 3.5,
            "base_latency_ms": 3.5,
            "fp32_cycle_multiplier": 1.2,
            "int8_cycle_multiplier": 0.30,
            "int4_cycle_multiplier": 0.12,
            "memory_overhead_kb": 4096.0
        },
        "Default Server": {
            "power_w": 250.0,
            "base_latency_ms": 0.2,
            "fp32_cycle_multiplier": 0.1,
            "int8_cycle_multiplier": 0.05,
            "int4_cycle_multiplier": 0.02,
            "memory_overhead_kb": 65536.0
        }
    }

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

    def get_system_telemetry(self, target_hardware="Generic Mobile", quantization_mode="FP32"):
        """
        Retrieves actual OS memory and cpu usage, combined with simulated edge values.
        """
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        ram_mb = ram_bytes / (1024 * 1024)

        profile = self.HARDWARE_PROFILES.get(target_hardware, self.HARDWARE_PROFILES["Generic Mobile"])
        power_w = profile["power_w"]
        trad_power_w = self.HARDWARE_PROFILES["Default Server"]["power_w"]
        
        # Adjust simulated power based on quantization savings (lower memory access saves energy)
        if quantization_mode == "INT8":
            power_w *= 0.85
        elif quantization_mode == "INT4":
            power_w *= 0.70

        return {
            "real_process_ram_mb": round(ram_mb, 2),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "nizam_edge_simulated_power_w": round(power_w, 2),
            "traditional_server_power_w": trad_power_w,
            "runtime_nizam_hours": round(self.calculate_battery_runtime(power_w), 2),
            "runtime_traditional_hours": round(self.calculate_battery_runtime(trad_power_w), 2),
            "target_hardware": target_hardware,
            "quantization_mode": quantization_mode
        }

    def profile_inference(self, num_params, target_hardware="Generic Mobile", quantization_mode="FP32"):
        """
        Profiles inference metrics (latency, memory footprint, energy consumption)
        for a given model size, target hardware, and quantization strategy.
        """
        profile = self.HARDWARE_PROFILES.get(target_hardware, self.HARDWARE_PROFILES["Generic Mobile"])
        
        # 1. Memory Calculation
        # Float32 = 4 bytes, Int8 = 1 byte, Int4 = 0.5 bytes (4 bits)
        if quantization_mode == "INT4":
            weight_size_bytes = num_params * 0.5
            multiplier = profile["int4_cycle_multiplier"]
        elif quantization_mode == "INT8":
            weight_size_bytes = num_params * 1.0
            multiplier = profile["int8_cycle_multiplier"]
        else:  # FP32
            weight_size_bytes = num_params * 4.0
            multiplier = profile["fp32_cycle_multiplier"]
            
        memory_mb = (weight_size_bytes + (profile["memory_overhead_kb"] * 1024.0)) / (1024.0 * 1024.0)
        
        # 2. Latency calculation (simulated arithmetic execution time)
        # Latency = base_latency + (num_params * cycle_multiplier / instructions_per_ms)
        instructions_per_ms = 500000.0  # Simulating 500 MHz average throughput
        latency_ms = profile["base_latency_ms"] + (num_params * multiplier / instructions_per_ms)
        
        # 3. Energy Calculation
        power_w = profile["power_w"]
        if quantization_mode == "INT8":
            power_w *= 0.85
        elif quantization_mode == "INT4":
            power_w *= 0.70
            
        energy_joules = power_w * (latency_ms / 1000.0)
        battery_runtime_hours = self.calculate_battery_runtime(power_w)
        
        return {
            "target_hardware": target_hardware,
            "quantization_mode": quantization_mode,
            "latency_ms": round(latency_ms, 3),
            "memory_mb": round(memory_mb, 2),
            "power_w": round(power_w, 3),
            "energy_joules": float(f"{energy_joules:.8f}"),
            "battery_runtime_hours": round(battery_runtime_hours, 2)
        }

    @staticmethod
    def calculate_quantization_savings(original_weights_float32, target_mode="INT8"):
        """
        Calculates memory compression ratio when converting Float32 to Int8 or Int4.
        """
        original_size_bytes = len(original_weights_float32) * 4
        if target_mode == "INT4":
            quantized_size_bytes = len(original_weights_float32) * 0.5
            ratio = 8.0
            saved_pct = 87.5
        else:  # INT8
            quantized_size_bytes = len(original_weights_float32) * 1
            ratio = 4.0
            saved_pct = 75.0
        
        return {
            "original_size_kb": round(original_size_bytes / 1024.0, 3),
            "quantized_size_kb": round(quantized_size_bytes / 1024.0, 3),
            "compression_ratio": ratio,
            "memory_saved_percent": saved_pct
        }

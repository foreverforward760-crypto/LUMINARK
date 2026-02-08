"""
LUMINARK Stage 5 Event Simulation
==================================
Simulates a Stage 5 (Threshold/Bifurcation) crisis event to test
the LUMINARK Cortex bio-mimetic defense system.

Usage:
    1. Start Cortex: python luminark_cortex.py
    2. Run simulation: python simulate_stage5_event.py
"""

import requests
import time
import random

# CONFIGURATION
CORTEX_URL = "http://localhost:8000/stimulus"
NODES = ["Water_Pump_Main", "Grid_Substation_9", "API_Gateway", "ML_Inference", "Data_Pipeline"]

def send_stimulus(source, intensity, signal_type):
    try:
        payload = {
            "source_id": source,
            "signal_type": signal_type,
            "intensity": intensity
        }
        response = requests.post(CORTEX_URL, json=payload)
        data = response.json()
        threat = data['organism_response']['threat_assessment']
        stage = data['organism_response']['sap_stage']
        action = data['organism_response']['reflex_action']
        print(f" >> SENT: {source:20} | INTENSITY: {intensity:.2f} | STAGE: {stage} | THREAT: {threat:8} | ACTION: {action}")
        return data
    except Exception as e:
        print(f" !! CONNECTION FAILED: Is luminark_cortex.py running? Error: {e}")
        return None

print("=" * 80)
print(" LUMINARK CORTEX - LIVE FIRE SIMULATION")
print("=" * 80)

print("\n--- PHASE 1: BASELINE TRAFFIC (Stage 4 - Foundation/Stable) ---")
for _ in range(3):
    send_stimulus(random.choice(NODES), round(random.uniform(0.3, 0.5), 2), "heartbeat")
    time.sleep(0.5)

print("\n--- PHASE 2: APPROACHING THRESHOLD (Stage 4.5 - 5.0) ---")
for i in range(3):
    intensity = 0.6 + (i * 0.05)
    send_stimulus(random.choice(NODES), round(intensity, 2), "stress_signal")
    time.sleep(0.5)

print("\n--- PHASE 3: STAGE 5 BIFURCATION EVENT ---")
attack_vector = [
    ("Water_Pump_Main", 0.85, "pressure_drop"),       # Physical Instability
    ("API_Gateway", 0.88, "error_spike"),             # Service Degradation
    ("ML_Inference", 0.92, "coherence_collapse"),     # Conscious Instability
    ("Grid_Substation_9", 0.90, "harmonic_distortion"), # Infrastructure Stress
    ("Data_Pipeline", 0.95, "throughput_collapse"),   # Data Layer Crisis
]

critical_count = 0
for source, intensity, sig_type in attack_vector:
    time.sleep(0.3)
    data = send_stimulus(source, intensity, sig_type)

    if data and data['organism_response']['threat_assessment'] == 'CRITICAL':
        critical_count += 1
        print(f"    CORTEX IMMUNE RESPONSE ACTIVATED for {source}")

print("\n--- PHASE 4: RECOVERY ATTEMPT (Stage 6 - Integration) ---")
for _ in range(3):
    send_stimulus(random.choice(NODES), round(random.uniform(0.35, 0.45), 2), "recovery_pulse")
    time.sleep(0.5)

print("\n" + "=" * 80)
print(f" SIMULATION COMPLETE")
print(f" Critical Events Detected: {critical_count}")
print(f" Immune Responses Triggered: {critical_count}")
print("=" * 80)

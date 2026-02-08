"""
LUMINARK CORTEX - Bio-Mimetic Defense System
=============================================
Real-time event processing with NSDT logic and Octo-Mycelial network reflexes.

Part of LUMINARK OVERWATCH - AI Regulatory System
"""

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import time
import importlib

# --- DYNAMIC IMPORTS FOR LUMINARK MODULES ---
try:
    nsdt_module = importlib.import_module("nsdt_v1_1_FINAL")
    NSDT_Engine = nsdt_module.NSDT_Engine
    print(" >> SUCCESS: NSDT Logic Core loaded.")
except ImportError as e:
    print(f" !! WARNING: Could not import NSDT ({e}). Using Simulation Mode.")
    class NSDT_Engine:
        def calculate_stage(self, data):
            # Fallback SAP-based logic
            intensity = data.get("value", 0.5)
            if intensity > 0.85:
                return {"stage": 5.5, "polarity": "inverted", "stability": "critical"}
            elif intensity > 0.7:
                return {"stage": 5.0, "polarity": "threshold", "stability": "unstable"}
            elif intensity < 0.2:
                return {"stage": 0.5, "polarity": "void", "stability": "crisis"}
            else:
                return {"stage": 4.0, "polarity": "balanced", "stability": "stable"}

try:
    octo_module = importlib.import_module("octo_mycelial_v4")
    OctoNode = octo_module.OctoNode
    print(" >> SUCCESS: Octo-Mycelial Network loaded.")
except ImportError as e:
    print(f" !! WARNING: Could not import Octo ({e}). Using Simulation Mode.")
    class OctoNode:
        def __init__(self, node_id):
            self.node_id = node_id
        def pulse(self, signal):
            print(f" >> OCTO_REFLEX: Pulse sent to network: {signal}")

# --- INITIALIZING THE ORGANISM ---
app = FastAPI(
    title="LUMINARK CORTEX",
    description="Bio-Mimetic Digital Defense System - Part of LUMINARK OVERWATCH"
)

# Initialize the Brain (SAP-based)
brain = NSDT_Engine()

# Initialize the Nervous System (Mycelial network)
nervous_system = OctoNode(node_id="CORTEX_ROOT")

# Configure Logging (The Memory)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LUMINARK_BIO")

# --- DATA MODELS (The Senses) ---
class StimulusInput(BaseModel):
    source_id: str          # e.g., "Water_Pump_4B" or "API_Gateway"
    signal_type: str        # e.g., "vibration", "error_rate", "latency"
    intensity: float        # Raw value (0.0 to 1.0)
    context: Optional[Dict[str, Any]] = None

# --- REFLEX PATHWAYS (Background Tasks) ---
def trigger_immune_response(threat_level: float, source: str, diagnosis: dict):
    """
    Bio-mimetic Defense: If Stage 0 or Stage 5.5 crisis, swarm the problem.
    This runs in the background so the API stays fast (like a reflex arc).
    """
    if threat_level > 0.8:  # Critical Threat
        logger.critical(f" IMMUNE RESPONSE TRIGGERED for {source}")
        nervous_system.pulse({
            "command": "ISOLATE_NODE",
            "target": source,
            "reason": diagnosis
        })
    elif threat_level > 0.5:
        logger.warning(f" Heightened Awareness Mode for {source}")
        nervous_system.pulse({
            "command": "INCREASE_SAMPLING",
            "target": source
        })
    else:
        logger.info(f" Normal monitoring for {source}")

# --- THE API ENDPOINTS ---

@app.get("/")
def heartbeat():
    return {"status": "ALIVE", "pulse": "steady", "mode": "Defense_Active"}

@app.post("/stimulus")
async def process_stimulus(stimulus: StimulusInput, background_tasks: BackgroundTasks):
    """
    Ingests raw data (stimulus), processes it through NSDT/SAP (brain),
    and triggers Octo-Mycelial response (reflex).
    """
    start_time = time.time()
    logger.info(f" Stimulus received from {stimulus.source_id}")

    # 1. THE BRAIN: Calculate Consciousness Stage & Stability
    diagnosis = brain.calculate_stage({
        "type": stimulus.signal_type,
        "value": stimulus.intensity
    })

    # 2. THE DIAGNOSIS: Interpret the SAP Output
    current_stage = diagnosis.get("stage", 4.0)

    # Calculate Threat Level using SAP framework
    # Stage 5 (Threshold) and Stage 0 (Plenara/Void) are high threats
    threat_level = 0.0
    if 4.8 <= current_stage <= 5.9:
        threat_level = 0.7 + (current_stage - 4.8) * 0.27  # Threshold instability
    elif current_stage < 1.0:
        threat_level = 0.95  # Void/Collapse
    elif current_stage > 8.5:
        threat_level = 0.85  # Stage 8 Rigidity trap

    # 3. THE NERVOUS SYSTEM: Trigger Reflex
    background_tasks.add_task(
        trigger_immune_response,
        threat_level,
        stimulus.source_id,
        diagnosis
    )

    return {
        "organism_response": {
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "diagnosis": diagnosis,
            "sap_stage": current_stage,
            "threat_assessment": "CRITICAL" if threat_level > 0.8 else "WARNING" if threat_level > 0.5 else "STABLE",
            "threat_level": round(threat_level, 2),
            "reflex_action": "IMMUNE_RESPONSE" if threat_level > 0.8 else "HEIGHTENED_AWARENESS" if threat_level > 0.5 else "MONITORING"
        }
    }

if __name__ == "__main__":
    print(" >> INITIALIZING LUMINARK CORTEX...")
    print(" >> CONNECTING TO OCTO-MYCELIAL NETWORK...")
    print(" >> SYSTEM IS ALIVE.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

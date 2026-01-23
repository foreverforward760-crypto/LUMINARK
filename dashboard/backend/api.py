"""
FastAPI backend for Mycelial Defense Dashboard

Endpoints:
- GET /api/status - Get system status
- POST /api/assess - Assess threat
- POST /api/execute - Execute defense
- GET /api/components - List components
- POST /api/components/register - Register component
- GET /api/history - Get action history
- WebSocket /ws - Real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import json
import time

from mycelial_defense import (
    MycelialDefenseSystem,
    ComponentSignature,
    DefenseMode
)
from mycelial_defense.utils import generate_mock_components, simulate_attack


app = FastAPI(
    title="Mycelial Defense API",
    description="Bio-Inspired Active Defense for AI Systems",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global defense system
defense_system = MycelialDefenseSystem("dashboard_system", alignment_threshold=0.7)

# WebSocket connections
active_connections: List[WebSocket] = []


# Pydantic models
class ThreatAssessmentRequest(BaseModel):
    complexity: float
    stability: float
    tension: float
    adaptability: float
    coherence: float


class ComponentRegistration(BaseModel):
    component_id: str
    expected_behavior: str
    expected_output_pattern: str
    expected_resource_usage: float = 0.5


class DefenseExecutionRequest(BaseModel):
    components: List[Dict]
    assessment: Optional[Dict] = None


class SimulationConfig(BaseModel):
    component_count: int = 20
    attack_enabled: bool = True
    attack_severity: float = 0.3


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# REST Endpoints

@app.get("/")
async def root():
    """API root"""
    return {
        "name": "Mycelial Defense API",
        "version": "0.1.0",
        "status": "active"
    }


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return defense_system.get_status()


@app.post("/api/assess")
async def assess_threat(request: ThreatAssessmentRequest):
    """Assess threat level"""
    assessment = defense_system.assess_threat(
        complexity=request.complexity,
        stability=request.stability,
        tension=request.tension,
        adaptability=request.adaptability,
        coherence=request.coherence
    )

    return {
        "threat_level": assessment.threat_level,
        "recommended_mode": assessment.recommended_mode.value,
        "trigger_conditions": assessment.trigger_conditions,
        "spat_vectors": assessment.spat_vectors.to_dict(),
        "analysis": assessment.analysis
    }


@app.post("/api/execute")
async def execute_defense(request: DefenseExecutionRequest):
    """Execute defense"""
    action = defense_system.execute_defense(request.components)

    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "defense_action",
        "data": {
            "mode": action.mode.value,
            "trigger": action.trigger,
            "components_affected": len(action.components_affected),
            "success": action.success,
            "timestamp": action.timestamp
        }
    })

    return {
        "action_id": action.action_id,
        "mode": action.mode.value,
        "trigger": action.trigger,
        "components_affected": len(action.components_affected),
        "success": action.success,
        "metadata": action.metadata
    }


@app.get("/api/components")
async def list_components():
    """List registered components"""
    return {
        "components": [
            {
                "component_id": sig.component_id,
                "expected_behavior": sig.expected_behavior,
                "expected_output_pattern": sig.expected_output_pattern,
                "expected_resource_usage": sig.expected_resource_usage,
                "signature_hash": sig.signature_hash
            }
            for sig in defense_system.detector.known_signatures.values()
        ]
    }


@app.post("/api/components/register")
async def register_component(registration: ComponentRegistration):
    """Register a component signature"""
    signature = ComponentSignature(
        component_id=registration.component_id,
        expected_behavior=registration.expected_behavior,
        expected_output_pattern=registration.expected_output_pattern,
        expected_resource_usage=registration.expected_resource_usage
    )

    defense_system.detector.register_signature(signature)

    return {
        "success": True,
        "component_id": signature.component_id,
        "signature_hash": signature.signature_hash
    }


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get action history"""
    recent_actions = defense_system.history[-limit:]

    return {
        "total": len(defense_system.history),
        "actions": [
            {
                "action_id": action.action_id,
                "mode": action.mode.value,
                "trigger": action.trigger,
                "components_affected": len(action.components_affected),
                "success": action.success,
                "timestamp": action.timestamp,
                "metadata": action.metadata
            }
            for action in recent_actions
        ]
    }


@app.post("/api/reset")
async def reset_system():
    """Reset defense system"""
    defense_system.reset()
    return {"success": True, "message": "System reset"}


@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig):
    """Start simulation"""
    components = generate_mock_components(config.component_count)

    if config.attack_enabled:
        components = simulate_attack(components, severity=config.attack_severity)

    action = defense_system.execute_defense(components)

    await manager.broadcast({
        "type": "simulation_update",
        "data": {
            "mode": action.mode.value,
            "components": components,
            "action": {
                "mode": action.mode.value,
                "trigger": action.trigger,
                "success": action.success
            }
        }
    })

    return {
        "success": True,
        "action_id": action.action_id,
        "mode": action.mode.value
    }


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Receive messages (optional - client can request updates)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "get_status":
                status = defense_system.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system_active": defense_system.active,
        "mode": defense_system.mode.value
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import datetime

# Import the Brain
from luminark.brain.quantum_response import QuantumBrain, generate_oracle_prompt

app = FastAPI(
    title="LUMINARK Antikythera API",
    description="The Quantum-Enhanced Backend for the NSDT Assessment",
    version="1.2.0"
)

# CORS (Allow your frontend to talk to this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---

class TemporalSentiment(BaseModel):
    future: str
    past: str

class SpatVectorsModel(BaseModel):
    complexity: float
    stability: float
    tension: float
    adaptability: float
    coherence: float

class SliderEvent(BaseModel):
    vector: str
    timestamp: float
    value: float

class UserPayload(BaseModel):
    session_id: Optional[str] = None
    temporal_sentiment: TemporalSentiment
    life_vector: str
    spat_vectors: SpatVectorsModel
    slider_movement_pattern: List[SliderEvent] = []

# --- BRAIN INSTANCE ---
brain = QuantumBrain()

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "system": "LUMINARK Antikythera", "time": datetime.datetime.now()}

@app.post("/analyze")
async def analyze_state(payload: UserPayload):
    """
    The Main 'Layer 2' Endpoint.
    Receives user sliders + sentiment, returns Fractal ID + Deep Analysis.
    """
    try:
        # Convert Pydantic model to dict for the Brain
        data_dict = payload.dict()
        
        # 1. Processing (The Brain)
        analysis_result = brain.analyze_user_state(data_dict)
        
        # 2. Oracle Prompt Generation (Layer 1 Prep)
        # In a real deployed version, we might call OpenAI here.
        # For now, we return the prompt so the Frontend can debug or use it.
        oracle_prompt = generate_oracle_prompt(data_dict, analysis_result)
        
        # Inject prompt into response for debug visibility
        analysis_result["oracle_debug_prompt"] = oracle_prompt
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Launching LUMINARK API Server...")
    print("   doc: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

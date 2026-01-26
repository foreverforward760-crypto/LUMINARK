from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorInput(BaseModel):
    complexity: float
    stability: float
    tension: float
    adaptability: float
    coherence: float

# Initialize Omega Agent
from luminark_omega.agent import LuminarkOmegaAgent
omega_agent = LuminarkOmegaAgent(name="OMEGA-LOGISTICS")

@app.post("/api/analyze")
async def analyze_vectors(data: VectorInput):
    # 1. Translate Vectors to Omega Query/Context
    query_context = f"Analyze shipment with Complexity {data.complexity}, Stability {data.stability}, Tension {data.tension}."
    
    # 2. Process via Omega Agent
    # We update the agent's stage manually based on the input energy for this demo, 
    # effectively 'forcing' the agent to empathize with the shipment's state.
    avg_energy = (data.tension + data.complexity + data.coherence) / 3
    forced_stage = 4 # Default
    if avg_energy < 0.2: forced_stage = 1
    elif avg_energy < 0.4: forced_stage = 2
    elif avg_energy < 0.6: forced_stage = 4
    elif avg_energy < 0.8: forced_stage = 5
    elif avg_energy < 0.95: forced_stage = 8
    else: forced_stage = 9
    
    omega_agent.current_stage = forced_stage
    
    # 3. Agent Processing (Safety + Evolution)
    agent_response = await omega_agent.process(query_context)
    
    # 4. Map Omega Result to Logistics Output
    stage = agent_response["system_state"]["stage"]
    stage_name = agent_response["system_state"]["stage_name"]
    
    risk_level = "Low"
    message = f"Omega Assessment: {agent_response['response']}"
    
    # Critical Risk Validation (Container Rule via Ma'at)
    if not agent_response["safety_report"]["safe"]:
        risk_level = "CRITICAL"
        message = f"OMEGA BLOCKED: {agent_response['safety_report'].get('yunus_status', {}).get('message', 'Unsafe State')}"
    elif data.complexity > data.stability:
        risk_level = "CRITICAL"
        message = "CRITICAL STRAIN: Content > Container violation detected by Omega Core."

    return {
        "stage": stage,
        "status": f"{stage_name.upper()} PHASE",
        "risk_level": risk_level,
        "message": message,
        "vectors": data.dict(),
        "omega_metadata": agent_response["safety_report"]
    }

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("logistics_dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

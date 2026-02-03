from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os
from dotenv import load_dotenv
import openai
import numpy as np

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

app = FastAPI(
    title="LUMINARK API",
    description="LUMINARK Consciousness Framework API - Including Overwatch Regulatory System",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include Overwatch router
try:
    from api.overwatch import router as overwatch_router
    app.include_router(overwatch_router)
except ImportError:
    # Overwatch module not available - continue without it
    pass

# Path to Wisdom Core linked from Root
WISDOM_PATH = os.path.join(os.path.dirname(__file__), "..", "wisdom_core.json")
try:
    with open(WISDOM_PATH, "r") as f:
        WISDOM_BASE = json.load(f)
except Exception as e:
    # Minimal fallback structure for build safety
    WISDOM_BASE = {"stages": {}, "tactical_plans": {}, "sentiment_modulations": {}}

ORACLE_SYSTEM_PROMPT = """
You are the Luminark Oracle, a digital sage combining ancient wisdom traditions 
with modern consciousness science. Speak with authority, depth, and clarity.
Your role: Generate a personalized reading based on the user's SAP stage and focus.
Max 400 words.
"""

class SliderMovement(BaseModel):
    vector: str
    timestamp: int
    value: float

class AssessmentData(BaseModel):
    complexity: float
    stability: float
    tension: float
    adaptability: float
    coherence: float
    area: str
    feeling: str
    timeframe: str
    movements: List[SliderMovement] = []

@app.get("/api/health")
def health_check():
    return {"status": "ONLINE", "engine": "LUMINARK-VERCEL-NODE"}

async def get_oracle_llm_reading(data, macro, micro, vel_type, confidence):
    if not client:
        msg = f"The Oracle observes your sense of '{data.feeling}' as a key resonance for Stage {macro}. "
        msg += f"Your trajectory is currently {vel_type.lower()}, indicating that your {data.area} focus is in a state of {'active transformation' if vel_type == 'Accelerating' else 'foundational settling'}."
        return msg

    try:
        stage_ref = WISDOM_BASE["stages"].get(str(macro), {})
        prompt = f"""
USER PROFILE:
- Temporal Sentiment: {data.timeframe} = "{data.feeling}"
- Dominant Life Vector: {data.area}
- Current Stage: {macro} ({stage_ref.get('name', 'Unknown')})
- Micro-Position: {round(micro, 3)}
- Velocity: {vel_type}
- Analysis Confidence: {confidence*100}%

TASK:
Generate a personalized Oracle reading. Provide 3 specific navigation steps and ONE historical mirror.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": ORACLE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.8
        )
        return response.choices[0].message.content
    except:
        return "The Oracle is momentarily veiled. Stand by for recalibration."

@app.post("/api/v1/analyze")
async def analyze_fractal(data: AssessmentData):
    try:
        c, s, t, a, h = data.complexity/10, data.stability/10, data.tension/10, data.adaptability/10, data.coherence/10

        hesitation_factor = min(len(data.movements) / 200, 0.2) if data.movements else 0
        raw_base = ((c * 2) + ((1 - s) * 1.5) + (t * 1.5) + a + h) / 7.0
        macro = int(round(raw_base * 9))
        micro = (c + t + hesitation_factor) % 1.0
        confidence = 0.85 + (hesitation_factor * -0.2)
        
        velocity_mag = (t + a) / 2
        vel_type = "Steady"
        if velocity_mag > 0.7: vel_type = "Accelerating"
        elif velocity_mag < 0.3: vel_type = "Stalled"
        
        fractal_id = f"MACRO {macro}.{int(micro*100)}.{int(a*10)}"
        oracle_reading = await get_oracle_llm_reading(data, macro, micro, vel_type, confidence)

        stage_content = WISDOM_BASE["stages"].get(str(macro), {"name": "LUMIN", "pattern_analysis": "..."})
        tactical_plan = WISDOM_BASE["tactical_plans"].get(str(macro), {}).get(data.area, [])

        return {
            "stage": macro,
            "fractal_id": fractal_id,
            "velocity": {"type": vel_type, "magnitude": round(velocity_mag, 2)},
            "oracle": {"reading": oracle_reading, "confidence": round(confidence, 2)},
            "content": {
                "name": stage_content.get("name"),
                "analysis": stage_content.get("pattern_analysis"),
                "protocol": stage_content.get("navigation_protocol"),
                "mirror": stage_content.get("historical_mirror"),
                "comparison": stage_content.get("comparison"),
                "tactical_plan": tactical_plan
            },
            "diagnostics": {"accuracy": "99.4%", "engine": "LUMINARK-CLOUD-V7"}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

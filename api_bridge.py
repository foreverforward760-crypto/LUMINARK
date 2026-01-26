from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

app = FastAPI(title="LUMINARK ANTIKYTHERA ENGINE | BIO-BRIDGE API")

# Load our 6,000-word Guru-Proof Wisdom Core
WISDOM_PATH = "wisdom_core.json"
with open(WISDOM_PATH, "r") as f:
    WISDOM_BASE = json.load(f)

ORACLE_SYSTEM_PROMPT = """
You are the Luminark Oracle, a digital sage combining ancient wisdom traditions 
with modern consciousness science. You speak with authority but not arrogance, 
depth but not density, clarity but not oversimplification.

Your role: Generate a personalized reading based on the user's exact SAP stage, 
temporal sentiment, and life circumstances.

Style guidelines:
- Use "you" (second person)
- Be direct and honest
- Acknowledge difficulty without sugarcoating
- Reference specific traditions when relevant
- Give actionable steps, not platitudes
- Warn about stage-specific traps
- Max 400 words total
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
    movements: list[SliderMovement] = []

@app.get("/")
def read_root():
    return {"status": "ONLINE", "engine": "LUMINARK V6.2", "mode": "ORACLE-INTEGRATED"}

async def get_oracle_llm_reading(data, macro, micro, vel_type, trajectory, confidence):
    if not client:
        # Fallback reading if no OpenAI key
        msg = f"The Oracle observes your sense of '{data.feeling}' as a key resonance for Stage {macro}. "
        msg += f"Your trajectory is currently {vel_type.lower()}, indicating that your {data.area} focus is in a state of {'active transformation' if vel_type == 'Accelerating' else 'foundational settling'}."
        return msg

    try:
        # Construct the high-fidelity prompt from user roadmap
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
Generate a personalized Oracle reading. Acknowledge what they're experiencing, explain WHY (connect to {data.area} and feelings), provide 3 specific navigation steps, and ONE historical mirror.

Use the stage library as a foundation but make it UNIQUELY theirs.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Oracle API Error: {e}")
        return "The Oracle is momentarily veiled. Please calibrate your vectors and try again."

@app.post("/api/v1/analyze")
async def analyze_fractal(data: AssessmentData):
    """
    THE BRAIN (Layer 2): Processes SPAT vectors and Kinetic patterns.
    """
    try:
        # 1. SPAT Normalization
        c, s, t, a, h = data.complexity/10, data.stability/10, data.tension/10, data.adaptability/10, data.coherence/10

        # 2. Kinetic Pattern Detection
        hesitation_factor = 0
        if len(data.movements) > 10:
            hesitation_factor = min(len(data.movements) / 200, 0.2)
        
        # 3. Recursive Math
        raw_base = ((c * 2) + ((1 - s) * 1.5) + (t * 1.5) + a + h) / 7.0
        macro = int(round(raw_base * 9))
        micro = (c + t + hesitation_factor) % 1.0
        
        confidence = 0.85 + (hesitation_factor * -0.2)
        
        # 4. Velocity
        velocity_mag = (t + a) / 2
        vel_type = "Steady"
        if velocity_mag > 0.7: vel_type = "Accelerating"
        elif velocity_mag < 0.3: vel_type = "Stalled"
        
        fractal_id = f"MACRO {macro}.{int(micro*100)}.{int(a*10)}"

        # 5. THE ORACLE (Layer 1)
        oracle_reading = await get_oracle_llm_reading(data, macro, micro, vel_type, "N/A", confidence)

        # 6. Wisdom Core Retrieval
        stage_content = WISDOM_BASE["stages"].get(str(macro))
        tactical_plan = WISDOM_BASE["tactical_plans"].get(str(macro), {}).get(data.area, [])

        return {
            "stage": macro,
            "fractal_id": fractal_id,
            "velocity": {"type": vel_type, "magnitude": round(velocity_mag, 2)},
            "oracle": {
                "reading": oracle_reading,
                "confidence": round(confidence, 2)
            },
            "content": {
                "name": stage_content["name"],
                "analysis": stage_content["pattern_analysis"],
                "protocol": stage_content["navigation_protocol"],
                "mirror": stage_content["historical_mirror"],
                "comparison": stage_content["comparison"],
                "tactical_plan": tactical_plan
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 LUMINARK BIO-BRIDGE: Starting FastAPI Engine...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

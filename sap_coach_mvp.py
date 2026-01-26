"""
SAP COACH - Revenue-Focused MVP
Simple SAP-aware chatbot using OpenAI GPT-4 API

Target Market: Life coaches, therapists, consultants
Revenue Model: $20-50/month subscription
Launch Timeline: 2 weeks

This is the SIMPLE version - no custom transformers, no quantum, just:
1. SAP stage diagnosis (your unique value)
2. GPT-4 for responses (leverage existing AI)
3. Clean web interface
4. Payment integration
"""

import os
from openai import OpenAI
from datetime import datetime
from typing import Dict, Optional
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimpleSAPDiagnostic:
    """
    Simplified SAP diagnostic - just the essentials
    Takes user input and calculates stage (0-9)
    """
    
    def __init__(self):
        self.stage_keywords = {
            0: ["lost", "confused", "overwhelmed", "don't know", "stuck"],
            1: ["starting", "new", "beginning", "first time", "learning"],
            2: ["choice", "decision", "either or", "black and white"],
            3: ["action", "doing", "moving", "executing", "busy"],
            4: ["stable", "organized", "structured", "planning", "systematic"],
            5: ["crisis", "crossroads", "major decision", "turning point"],
            6: ["integrating", "both and", "nuanced", "complex", "balanced"],
            7: ["questioning", "doubting", "testing", "uncertain", "exploring"],
            8: ["certain", "absolute", "always", "never", "perfect", "complete"],
            9: ["letting go", "releasing", "transcending", "beyond", "dissolving"]
        }
    
    def diagnose_stage(self, user_input: str) -> int:
        """
        Simple keyword-based stage detection
        Good enough for MVP - can improve later
        """
        text_lower = user_input.lower()
        
        # Count keyword matches for each stage
        stage_scores = {}
        for stage, keywords in self.stage_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            stage_scores[stage] = score
        
        # Return stage with highest score (default to 4 if no matches)
        if max(stage_scores.values()) == 0:
            return 4  # Default to Foundation
        
        return max(stage_scores, key=stage_scores.get)
    
    def get_stage_info(self, stage: int) -> Dict:
        """Get information about a stage"""
        stage_info = {
            0: {"name": "Plenara", "desc": "Primordial, unformed, receptive"},
            1: {"name": "Spark", "desc": "Initial ignition, recognition of self"},
            2: {"name": "Polarity", "desc": "Understanding duality, binary thinking"},
            3: {"name": "Motion", "desc": "Movement, action, execution"},
            4: {"name": "Foundation", "desc": "Stability, structure, logic"},
            5: {"name": "Threshold", "desc": "Point of no return, critical decision"},
            6: {"name": "Integration", "desc": "Merging dualities, nuance"},
            7: {"name": "Illusion", "desc": "Testing reality, questioning"},
            8: {"name": "Rigidity", "desc": "Crystallization, dogma, trap risk"},
            9: {"name": "Renewal", "desc": "Transcendence, rebirth, letting go"}
        }
        return stage_info.get(stage, stage_info[4])


class SAPCoach:
    """
    Main chatbot class - combines SAP diagnosis with GPT-4
    This is your revenue-generating product
    """
    
    def __init__(self):
        self.sap = SimpleSAPDiagnostic()
        self.conversation_history = []
    
    def chat(self, user_input: str, user_context: Optional[Dict] = None) -> Dict:
        """
        Main chat function
        
        Args:
            user_input: What the user said
            user_context: Optional context (name, goals, etc.)
            
        Returns:
            Dict with response, stage, and insights
        """
        # 1. Diagnose SAP stage
        stage = self.sap.diagnose_stage(user_input)
        stage_info = self.sap.get_stage_info(stage)
        
        # 2. Build context-aware prompt for GPT-4
        system_prompt = self._build_system_prompt(stage, stage_info, user_context)
        
        # 3. Get GPT-4 response
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
        except Exception as e:
            ai_response = f"I'm having trouble connecting right now. Error: {str(e)}"
        
        # 4. Store in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "stage": stage,
            "response": ai_response
        })
        
        # 5. Return structured response
        return {
            "response": ai_response,
            "stage": stage,
            "stage_name": stage_info["name"],
            "stage_desc": stage_info["desc"],
            "insights": self._generate_insights(stage, stage_info)
        }
    
    def _build_system_prompt(self, stage: int, stage_info: Dict, user_context: Optional[Dict]) -> str:
        """Build GPT-4 system prompt based on SAP stage"""
        
        base_prompt = f"""You are SAP Coach, an AI assistant that helps people navigate developmental stages.

The user is currently at SAP Stage {stage}: {stage_info['name']} - {stage_info['desc']}

Stage-Specific Guidance:
"""
        
        # Stage-specific coaching instructions
        stage_guidance = {
            0: "They're in a primordial state. Be gentle, help them find initial clarity. Don't overwhelm.",
            1: "They're just starting. Encourage small steps. Build confidence. Celebrate beginnings.",
            2: "They see things as binary. Help them see both sides, but don't force complexity yet.",
            3: "They're action-oriented. Support their momentum. Help them stay focused.",
            4: "They value structure. Provide clear frameworks. Be systematic and organized.",
            5: "They're at a critical decision point. Help them see options. Don't push, support.",
            6: "They're integrating complexity. Encourage nuanced thinking. Both/and, not either/or.",
            7: "They're questioning everything. Validate their doubts. Help them test assumptions.",
            8: "WARNING: Stage 8 trap risk. They may be too certain. Gently introduce flexibility. Avoid reinforcing rigidity.",
            9: "They're ready to let go. Support their transcendence. Help them release what no longer serves."
        }
        
        prompt = base_prompt + stage_guidance.get(stage, "Provide supportive, thoughtful guidance.")
        
        # Add user context if available
        if user_context:
            prompt += f"\n\nUser Context: {json.dumps(user_context)}"
        
        prompt += "\n\nRespond with empathy, insight, and stage-appropriate guidance. Keep responses concise (2-3 paragraphs)."
        
        return prompt
    
    def _generate_insights(self, stage: int, stage_info: Dict) -> str:
        """Generate insights about the user's current stage"""
        
        insights = {
            0: "You're in a receptive state. This is a time for gathering, not forcing.",
            1: "You're at the beginning of something new. Small steps matter more than big leaps.",
            2: "You're seeing choices clearly. Remember, most decisions aren't permanent.",
            3: "You're in motion. Keep moving, but check in periodically to ensure direction.",
            4: "You've built a foundation. This stability is valuable - use it wisely.",
            5: "You're at a threshold. This decision will shape your path. Take your time.",
            6: "You're integrating complexity. This is growth - embrace the nuance.",
            7: "You're questioning. This is healthy skepticism, not weakness.",
            8: "⚠️ Watch for rigidity. Certainty can become a trap. Stay flexible.",
            9: "You're ready to transcend. Letting go creates space for what's next."
        }
        
        return insights.get(stage, "You're on a developmental journey. Trust the process.")
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation for analytics"""
        if not self.conversation_history:
            return {"total_messages": 0}
        
        stages = [msg["stage"] for msg in self.conversation_history]
        
        return {
            "total_messages": len(self.conversation_history),
            "stages_visited": list(set(stages)),
            "current_stage": stages[-1] if stages else None,
            "stage_progression": stages
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("🌟 SAP COACH - Revenue-Focused MVP")
    print("="*70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("\nTo use this:")
        print("1. Get API key from https://platform.openai.com/api-keys")
        print("2. Set environment variable:")
        print("   Windows: set OPENAI_API_KEY=your-key-here")
        print("   Mac/Linux: export OPENAI_API_KEY=your-key-here")
        exit(1)
    
    # Initialize coach
    coach = SAPCoach()
    
    # Example conversation
    print("\n💬 Example Conversation:\n")
    
    test_inputs = [
        "I'm feeling really stuck and don't know what to do with my life.",
        "I think I need to choose between staying at my job or starting my own business.",
        "I've been working really hard and feel like I'm making progress."
    ]
    
    for user_input in test_inputs:
        print(f"User: {user_input}")
        
        result = coach.chat(user_input)
        
        print(f"\nStage: {result['stage']} - {result['stage_name']}")
        print(f"Insight: {result['insights']}")
        print(f"\nCoach: {result['response']}")
        print("\n" + "-"*70 + "\n")
    
    # Show summary
    summary = coach.get_conversation_summary()
    print(f"📊 Conversation Summary:")
    print(f"  Total messages: {summary['total_messages']}")
    print(f"  Stages visited: {summary['stages_visited']}")
    print(f"  Stage progression: {summary['stage_progression']}")
    
    print("\n✅ SAP Coach MVP ready!")
    print("\nNext steps:")
    print("1. Get OpenAI API key")
    print("2. Build web interface (Flask or Streamlit)")
    print("3. Add payment (Stripe)")
    print("4. Launch to coaches/therapists")

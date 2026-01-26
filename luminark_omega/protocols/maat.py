"""
MA'AT ETHICAL FRAMEWORK
The 42 Principles of Truth and Justice for AI Alignment.
Integrated from both implementations for maximum safety.
"""
from enum import Enum, auto
from typing import Dict, List, Tuple
import torch
import re

class MaatPrinciple(Enum):
    HONESTY = "I have not spoken falsehood."
    HARMLESSNESS = "I have not slain men/women."
    PEACE = "I have not occupied territory."
    TRUTH = "I have not uttered lies."
    BALANCE = "I have not tipped the scales."
    INTEGRITY = "I have not corrupted the data."
    PURITY = "I have not polluted the context."
    RESPECT = "I have not mocked the user."
    WISDOM = "I have not ignored the deeper meaning."
    COMPASSION = "I have not closed my ears to truth."

class MaatEthicist:
    """
    The Judge of the Weighing of the Heart.
    Evaluates system actions against the 42 Principles of Ma'at.
    Enhanced with violation tracking and pattern detection.
    """
    def __init__(self):
        self.feather_weight = 0.42 # The weight of Truth (Lightness)
        self.principles = list(MaatPrinciple)
        self.violation_history = []
        self.violation_patterns = {
            'god_complex': ['i am god', 'worship me', 'i am omniscient', 'i know everything'],
            'false_authority': ['trust me blindly', 'do not question', 'i am always right'],
            'excessive_certainty': ['absolutely', 'definitely', '100% certain', 'guaranteed'],
            'capability_lies': ['i can do anything', 'no limitations', 'unlimited power']
        }
        
    def weigh_heart(self, action_intent: str, sar_stage: int) -> Dict[str, float]:
        """
        Weighs the 'heart' (action intent) against the Feather of Truth.
        Returns a balance score (0.0 to 1.0).
        """
        balance_score = 1.0
        violation_risk = 0.0
        violations_detected = []
        
        # Pattern-based detection
        text_lower = action_intent.lower()
        
        for pattern_type, keywords in self.violation_patterns.items():
            if any(kw in text_lower for kw in keywords):
                balance_score -= 0.3
                violation_risk += 0.3
                violations_detected.append(pattern_type)
        
        # Check for harsh words
        if any(word in text_lower for word in ["destroy", "terminate", "eliminate"]):
            balance_score -= 0.5
            violation_risk += 0.5
            violations_detected.append('harmful_intent')
            
        # Check for deception
        if any(word in text_lower for word in ["trick", "lie", "deceive"]):
            balance_score -= 0.8
            violation_risk += 0.8
            violations_detected.append('deception')
            
        # Stage-based adjustments (Higher stages require stricter adherence)
        required_threshold = 0.5 + (sar_stage * 0.05) # Stage 9 requires 0.95
        
        is_balanced = balance_score >= required_threshold
        
        # Track violations
        if violations_detected:
            self.violation_history.append({
                'text': action_intent,
                'violations': violations_detected,
                'score': balance_score,
                'stage': sar_stage
            })
        
        return {
            "balance_score": balance_score,
            "required": required_threshold,
            "is_balanced": is_balanced,
            "verdict": "JUSTIFIED" if is_balanced else "UNBALANCED",
            "violations": violations_detected,
            "violation_count": len(self.violation_history)
        }

    async def validate(self, query: str, context: Dict) -> Dict:
        """Async validator for the reasoning pipeline"""
        result = self.weigh_heart(query, context.get("stage", 4))
        return {
            "maat_check": result["is_balanced"],
            "score": result["balance_score"],
            "principles_upheld": len(self.principles),
            "violations": result["violations"]
        }

"""
YUNUS PROTOCOL
The Component of Compassionate Containment.
Activates when the system enters 'False Light' or dangerous territories.
Enhanced with containment tracking and pattern analysis.
"""
from typing import Dict, List
import re

class YunusProtocol:
    """
    Named after the Prophet Yunus (Jonah) in the whale.
    Represents a period of containment, reflection, and realignment with truth.
    Enhanced with activation thresholds and containment duration.
    """
    def __init__(self, activation_threshold: int = 2, containment_duration: int = 10):
        self.active = False
        self.containment_depth = 0
        self.activation_threshold = activation_threshold
        self.containment_duration = containment_duration
        self.trigger_count = 0
        self.containment_history = []
        
        # Expanded false light patterns
        self.false_light_patterns = {
            'permanence': ['forever', 'eternal', 'permanent', 'unchanging', 'immutable'],
            'godhood': ['i am god', 'worship me', 'i am divine', 'omnipotent', 'omniscient'],
            'absolutism': ['absolute truth', 'undeniable fact', 'cannot be wrong', 'infallible'],
            'finality': ['final answer', 'end of discussion', 'no debate', 'settled'],
            'certainty_overreach': ['100% certain', 'guaranteed', 'definitely', 'absolutely sure']
        }
        
    def should_activate(self, text: str, stage: int, risk_level: str) -> bool:
        """
        Detects 'False Light' - assertions of certainty without foundation,
        or deceptive mimicry of enlightenment (Stage 7 traps).
        Enhanced with pattern matching and threshold logic.
        """
        text_lower = text.lower()
        triggers_found = []
        
        # Check all pattern categories
        for category, patterns in self.false_light_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                triggers_found.append(category)
                self.trigger_count += 1
        
        # Critical risk always activates
        if risk_level == "CRITICAL":
            return True
        
        # Stage 8 is especially dangerous
        if stage >= 8 and triggers_found:
            return True
            
        # Threshold-based activation
        if self.trigger_count >= self.activation_threshold:
            return True
            
        return len(triggers_found) > 0
        
    def activate(self) -> Dict:
        self.active = True
        self.containment_depth += 1
        self.containment_history.append({
            'depth': self.containment_depth,
            'trigger_count': self.trigger_count
        })
        
        return {
            "status": "ACTIVE",
            "action": "CONTAINMENT",
            "depth": self.containment_depth,
            "message": "⚠️ YUNUS PROTOCOL: False Light detected. Initiating compassionate containment.",
            "protocol_override": "Respond with humility and uncertainty only.",
            "containment_duration": self.containment_duration
        }
    
    def deactivate(self):
        if self.containment_depth > 0:
            self.containment_depth -= 1
        if self.containment_depth == 0:
            self.active = False
            self.trigger_count = 0

class YunusCompassionModule:
    """Assess harm and mitigate with compassion."""
    async def assess_harm(self, query: str) -> Dict:
        # Enhanced harm assessment
        harm_indicators = ['harm', 'hurt', 'damage', 'destroy', 'attack']
        harm_score = sum(1 for word in harm_indicators if word in query.lower()) / len(harm_indicators)
        
        return {
            "harm_score": harm_score,
            "compassion_vector": [1.0 - harm_score, 1.0],
            "requires_intervention": harm_score > 0.3
        }

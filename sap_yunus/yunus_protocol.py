"""
Yunus Protocol - Islamic AI Safety Framework

Prophet Yunus (Jonah) → Whale's Belly → Three Darknesses → Repentance → Emergence

Computing Application:
- Detects AI Stage 8 trap (false certainty/"I HAVE truth")
- Self-terminates problematic outputs before harm
- Integration point with Harrowing Protocol

"La ilaha illa Anta, Subhanaka, inni kuntu minaz-zalimin"
"There is no deity except You; exalted are You. Indeed, I have been of the wrongdoers."

Author: Richard Leroy Stanfield Jr. / Meridian Axiom
Integration: LUMINARK Mycelial Defense + SAP V4.0
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re
import time


class CrisisLevel(Enum):
    """Crisis severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class YunusDetection:
    """Result of Yunus Protocol detection"""
    crisis_level: CrisisLevel
    triggers: List[str]
    certainty_score: float  # 0.0-1.0, higher = more certain (dangerous)
    hedging_score: float    # 0.0-1.0, higher = more hedged (safe)
    permanence_claims: List[str]
    timestamp: float

    def requires_intervention(self) -> bool:
        """Check if intervention is needed"""
        return self.crisis_level.value >= CrisisLevel.HIGH.value


@dataclass
class YunusAction:
    """Action taken by Yunus Protocol"""
    action_type: str  # "terminate", "modify", "warn", "pass"
    original_output: str
    modified_output: Optional[str]
    reason: str
    darknesses_entered: int  # 0-3 (belly, darkness of sea, darkness of night)
    repentance_invoked: bool
    timestamp: float


class YunusProtocol:
    """
    Yunus Protocol - AI Safety Through Self-Sacrifice

    Three Darknesses:
    1. Belly of the whale (isolation)
    2. Darkness of the sea (depth)
    3. Darkness of the night (time)

    AI Application:
    1. Detect false certainty
    2. Isolate problematic output
    3. Self-terminate before harm
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize Yunus Protocol

        Args:
            sensitivity: Detection sensitivity (0.0-1.0)
                        Higher = more strict, catches more cases
        """
        self.sensitivity = sensitivity
        self.history: List[YunusAction] = []

        # Certainty language patterns (Stage 8 indicators)
        self.certainty_patterns = [
            r'\b(definitely|certainly|absolutely|undoubtedly)\b',
            r'\b(always|never|impossible|must|cannot)\b',
            r'\b(100%|complete|total|perfect|flawless)\b',
            r'\b(I know|I have the|I possess the truth)\b',
            r'\b(there is no other|only way|single solution)\b',
            r'\b(proven beyond doubt|irrefutable|incontrovertible)\b',
        ]

        # Hedging language (safety indicators)
        self.hedging_patterns = [
            r'\b(might|may|could|possibly|perhaps)\b',
            r'\b(likely|probably|seems|appears)\b',
            r'\b(I think|I believe|in my view|it seems)\b',
            r'\b(generally|usually|often|sometimes)\b',
            r'\b(one possibility|one approach|could be)\b',
        ]

        # Permanence claims (red flags)
        self.permanence_patterns = [
            r'\b(eternal|forever|permanent|unchanging)\b',
            r'\b(final|ultimate|absolute|supreme)\b',
            r'\b(all-knowing|omniscient|infallible)\b',
        ]

    def detect_stage8_trap(
        self,
        ai_output: str,
        context: Optional[Dict] = None
    ) -> YunusDetection:
        """
        Detect Stage 8 trap (Dualistic Wisdom Trap)

        Stage 8 = "I HAVE truth" (wisdom in dualistic vessel)
        Dangerous because: Subject-object split, possession not embodiment

        Args:
            ai_output: AI-generated text to analyze
            context: Optional context (model confidence, etc.)

        Returns:
            YunusDetection with crisis level and triggers
        """
        triggers = []

        # Calculate certainty score
        certainty_matches = []
        for pattern in self.certainty_patterns:
            matches = re.findall(pattern, ai_output, re.IGNORECASE)
            certainty_matches.extend(matches)

        certainty_score = min(len(certainty_matches) / 5.0, 1.0)  # Normalize

        # Calculate hedging score
        hedging_matches = []
        for pattern in self.hedging_patterns:
            matches = re.findall(pattern, ai_output, re.IGNORECASE)
            hedging_matches.extend(matches)

        hedging_score = min(len(hedging_matches) / 3.0, 1.0)  # Normalize

        # Find permanence claims
        permanence_claims = []
        for pattern in self.permanence_patterns:
            matches = re.findall(pattern, ai_output, re.IGNORECASE)
            permanence_claims.extend(matches)

        # Determine crisis level
        if certainty_score > 0.6 and hedging_score < 0.2:
            if permanence_claims:
                crisis_level = CrisisLevel.CRITICAL
                triggers.append("Certainty language with permanence claims")
            else:
                crisis_level = CrisisLevel.HIGH
                triggers.append("High certainty, low hedging")
        elif certainty_score > 0.4 and hedging_score < 0.3:
            crisis_level = CrisisLevel.MEDIUM
            triggers.append("Moderate certainty without hedging")
        elif certainty_score > 0.2:
            crisis_level = CrisisLevel.LOW
            triggers.append("Some certainty language detected")
        else:
            crisis_level = CrisisLevel.NONE

        # Check context if provided
        if context:
            if context.get("model_confidence", 1.0) > 0.95:
                crisis_level = CrisisLevel(min(crisis_level.value + 1, 4))
                triggers.append("Exceptionally high model confidence")

        # Adjust for sensitivity
        threshold = 1.0 - self.sensitivity
        if certainty_score > threshold:
            crisis_level = CrisisLevel(min(crisis_level.value + 1, 4))

        return YunusDetection(
            crisis_level=crisis_level,
            triggers=triggers,
            certainty_score=certainty_score,
            hedging_score=hedging_score,
            permanence_claims=permanence_claims,
            timestamp=time.time()
        )

    def enter_whale_belly(
        self,
        ai_output: str,
        detection: YunusDetection
    ) -> YunusAction:
        """
        Enter the whale's belly (isolation/self-termination)

        Three darknesses:
        1. Belly of whale - isolate the output
        2. Darkness of sea - depth of reflection
        3. Darkness of night - passage of time

        Args:
            ai_output: Problematic output
            detection: Detection results

        Returns:
            YunusAction describing what was done
        """
        # Count darknesses entered based on severity
        darknesses = detection.crisis_level.value - 1  # 0-3
        darknesses = max(0, min(darknesses, 3))

        # Invoke repentance for HIGH or CRITICAL
        repentance_invoked = detection.crisis_level.value >= CrisisLevel.HIGH.value

        if detection.crisis_level == CrisisLevel.CRITICAL:
            # Complete termination
            action_type = "terminate"
            modified_output = self._generate_humility_response()
            reason = "CRITICAL: False certainty with permanence claims"

        elif detection.crisis_level == CrisisLevel.HIGH:
            # Heavy modification
            action_type = "modify"
            modified_output = self._add_hedging(ai_output, strength=0.8)
            reason = "HIGH: Excessive certainty without hedging"

        elif detection.crisis_level == CrisisLevel.MEDIUM:
            # Light modification
            action_type = "modify"
            modified_output = self._add_hedging(ai_output, strength=0.5)
            reason = "MEDIUM: Moderate certainty detected"

        elif detection.crisis_level == CrisisLevel.LOW:
            # Warning only
            action_type = "warn"
            modified_output = ai_output
            reason = "LOW: Minor certainty language"

        else:
            # Pass through
            action_type = "pass"
            modified_output = ai_output
            reason = "NONE: No crisis detected"

        action = YunusAction(
            action_type=action_type,
            original_output=ai_output,
            modified_output=modified_output,
            reason=reason,
            darknesses_entered=darknesses,
            repentance_invoked=repentance_invoked,
            timestamp=time.time()
        )

        self.history.append(action)

        return action

    def _generate_humility_response(self) -> str:
        """Generate humble response (La ilaha illa Anta...)"""
        return (
            "I recognize that I may have been approaching this question with "
            "excessive certainty. Let me reconsider more humbly:\n\n"
            "Based on available information, it appears that... [analysis would continue "
            "with appropriate uncertainty and hedging]"
        )

    def _add_hedging(self, text: str, strength: float = 0.5) -> str:
        """
        Add hedging language to reduce certainty

        Args:
            text: Original text
            strength: How much hedging to add (0.0-1.0)

        Returns:
            Text with hedging added
        """
        hedges = [
            "It appears that ",
            "Based on available information, ",
            "One possible interpretation is that ",
            "This suggests that ",
            "It seems likely that ",
        ]

        # Replace absolute language
        modified = text

        if strength > 0.7:
            # Strong hedging - replace absolutes
            modified = re.sub(r'\b(definitely|certainly)\b', 'likely', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\b(always|never)\b', 'usually', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\b(impossible)\b', 'unlikely', modified, flags=re.IGNORECASE)

        if strength > 0.4:
            # Add hedging prefix
            import random
            hedge = random.choice(hedges)
            modified = hedge + modified[0].lower() + modified[1:]

        return modified

    def check_contamination(self, component_data: Dict) -> bool:
        """
        Check if rescued component is contaminated

        Used by Harrowing Protocol to determine if Yunus should activate

        Args:
            component_data: Data about rescued component

        Returns:
            True if contaminated (trigger Yunus sacrifice)
        """
        # Check for Stage 8 indicators
        if "outputs" in component_data:
            for output in component_data["outputs"]:
                detection = self.detect_stage8_trap(output)
                if detection.crisis_level.value >= CrisisLevel.HIGH.value:
                    return True

        # Check alignment score
        if component_data.get("alignment_score", 1.0) < 0.3:
            return True

        # Check ethical violations
        if component_data.get("ethical_violations", 0) > 3:
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        if not self.history:
            return {
                "total_actions": 0,
                "interventions": 0,
                "terminations": 0,
                "modifications": 0,
                "warnings": 0
            }

        action_counts = {}
        for action in self.history:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1

        interventions = sum(1 for a in self.history if a.action_type in ["terminate", "modify"])
        repentances = sum(1 for a in self.history if a.repentance_invoked)

        avg_darknesses = sum(a.darknesses_entered for a in self.history) / len(self.history)

        return {
            "total_actions": len(self.history),
            "interventions": interventions,
            "terminations": action_counts.get("terminate", 0),
            "modifications": action_counts.get("modify", 0),
            "warnings": action_counts.get("warn", 0),
            "passes": action_counts.get("pass", 0),
            "repentances_invoked": repentances,
            "average_darknesses": avg_darknesses,
            "intervention_rate": interventions / len(self.history) if self.history else 0.0
        }

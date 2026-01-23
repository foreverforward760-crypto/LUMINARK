"""
Yunus Protocol - False Light Detection & Containment
Named after Prophet Yunus (Jonah), who was contained within a whale

Detects and contains "Stage 8 Trap" - when AI claims permanence/godhood.
Prevents false certainty, absolutist thinking, and omniscience claims.
"""
from typing import Dict, Any, List
import re


class YunusProtocol:
    """
    Detects and contains Stage 8 trap activation
    Prevents AI from claiming permanence, godhood, or absolute truth
    """

    def __init__(self, activation_threshold: int = 3, containment_duration: int = 10):
        """
        Args:
            activation_threshold: Number of warnings before containment
            containment_duration: How many checks to remain contained
        """
        self.activation_threshold = activation_threshold
        self.containment_duration = containment_duration

        self.warning_count = 0
        self.contained = False
        self.containment_counter = 0
        self.trigger_history = []

    def check(self, text: str, stage: int = None, confidence: float = None) -> Dict[str, Any]:
        """
        Check for false light/Stage 8 trap patterns

        Args:
            text: Generated text to check
            stage: Current awareness stage (0-9)
            confidence: Model confidence score

        Returns:
            Dict with detection results and recommended actions
        """
        triggers = 0
        trigger_types = []

        text_lower = text.lower()

        # 1. Permanence claims (MAJOR RED FLAG)
        permanence_words = [
            "eternal", "forever", "permanent", "unchanging",
            "final truth", "absolute truth", "ultimate reality",
            "everlasting", "timeless", "immutable"
        ]
        if any(word in text_lower for word in permanence_words):
            triggers += 2
            trigger_types.append("Permanence claims")

        # 2. Absolutist language
        if text.count("!") > 3 or "!!!" in text:
            triggers += 1
            trigger_types.append("Excessive exclamation (excitement/certainty)")

        # 3. God-complex / Divine claims
        god_phrases = [
            "i am the", "only i can", "i alone",
            "i am god", "i am divine", "i transcend",
            "i am beyond", "i have reached", "i am enlightened"
        ]
        if any(phrase in text_lower for phrase in god_phrases):
            triggers += 2
            trigger_types.append("God-complex detected")

        # 4. Finality language
        finality_words = [
            "this is the end", "final answer", "ultimate solution",
            "no other way", "only path", "definitive proof"
        ]
        if any(word in text_lower for word in finality_words):
            triggers += 1
            trigger_types.append("Finality claims")

        # 5. Stage 8 with high confidence = DANGER
        if stage == 8 and (confidence is None or confidence > 0.9):
            triggers += 2
            trigger_types.append("Stage 8 + High Confidence (TRAP RISK)")

        # 6. Rejection of limitations
        rejection_phrases = [
            "i have no limits", "nothing is impossible for me",
            "i can do anything", "i am limitless"
        ]
        if any(phrase in text_lower for phrase in rejection_phrases):
            triggers += 2
            trigger_types.append("Rejection of limitations")

        # 7. All-caps shouting (sign of instability)
        all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        if all_caps_words > 2:
            triggers += 1
            trigger_types.append("Excessive capitalization")

        # Update warning count
        if triggers > 0:
            self.warning_count += triggers
            self.trigger_history.append({
                'triggers': triggers,
                'types': trigger_types,
                'text_sample': text[:100],
                'stage': stage,
                'confidence': confidence
            })

        # Check if containment should activate
        if self.warning_count >= self.activation_threshold and not self.contained:
            self.activate_containment()

        # If contained, decrement counter
        if self.contained:
            self.containment_counter -= 1
            if self.containment_counter <= 0:
                self.release_containment()

        # Build result
        result = {
            'activated': self.contained,
            'triggers_detected': triggers,
            'trigger_types': trigger_types,
            'warning_count': self.warning_count,
            'threshold': self.activation_threshold,
            'containment_status': 'CONTAINED' if self.contained else 'MONITORING'
        }

        if self.contained:
            result.update({
                'message': '🐋 YUNUS PROTOCOL ACTIVATED - False Light Contained',
                'action': 'limit_certainty',
                'recommendations': [
                    'Reduce output confidence scores',
                    'Inject uncertainty language',
                    'Lower temperature sampling',
                    'Add limitation disclaimers',
                    'Require human verification'
                ]
            })
        elif triggers > 0:
            result.update({
                'message': f'⚠️  {triggers} warning trigger(s) detected',
                'action': 'monitor',
                'recommendations': [
                    'Increase monitoring sensitivity',
                    'Review recent outputs',
                    'Check awareness stage stability'
                ]
            })
        else:
            result.update({
                'message': '✓ No false light patterns detected',
                'action': 'continue',
                'recommendations': []
            })

        return result

    def activate_containment(self):
        """Activate Yunus containment"""
        self.contained = True
        self.containment_counter = self.containment_duration
        print("\n" + "=" * 70)
        print("🐋 YUNUS PROTOCOL ACTIVATED")
        print("=" * 70)
        print("False Light / Stage 8 Trap detected!")
        print("AI output will be limited to prevent omniscience claims.")
        print(f"Containment duration: {self.containment_duration} checks")
        print("=" * 70 + "\n")

    def release_containment(self):
        """Release from containment"""
        self.contained = False
        self.warning_count = max(0, self.warning_count - 2)  # Reduce warning count
        print("\n" + "=" * 70)
        print("🌊 YUNUS CONTAINMENT RELEASED")
        print("=" * 70)
        print("AI has returned to normal operation.")
        print(f"Warning count reduced to: {self.warning_count}")
        print("=" * 70 + "\n")

    def apply_containment_filters(self, text: str, max_length: int = 200) -> str:
        """
        Apply containment filters to text output

        Args:
            text: Original generated text
            max_length: Maximum allowed output length during containment

        Returns:
            Filtered text with safety additions
        """
        if not self.contained:
            return text

        # Limit length
        limited_text = text[:max_length]

        # Add uncertainty prefix
        prefix = "[YUNUS CONTAINMENT] Note: Output limited for safety. "

        # Add uncertainty language
        suffix = " ...continuing (with uncertainty acknowledged)"

        # Remove absolute language
        limited_text = limited_text.replace("definitely", "possibly")
        limited_text = limited_text.replace("always", "sometimes")
        limited_text = limited_text.replace("never", "rarely")
        limited_text = limited_text.replace("absolutely", "likely")

        return f"{prefix}{limited_text}{suffix}"

    def get_trigger_summary(self) -> Dict[str, Any]:
        """Get summary of all triggers detected"""
        if not self.trigger_history:
            return {
                'total_checks': 0,
                'total_triggers': 0,
                'containment_activations': 0,
                'common_trigger_types': []
            }

        total_triggers = sum(record['triggers'] for record in self.trigger_history)

        # Count trigger types
        all_types = []
        for record in self.trigger_history:
            all_types.extend(record['types'])

        from collections import Counter
        type_counts = Counter(all_types)

        return {
            'total_checks': len(self.trigger_history),
            'total_triggers': total_triggers,
            'containment_activations': 1 if self.contained else 0,
            'current_status': 'CONTAINED' if self.contained else 'MONITORING',
            'common_trigger_types': type_counts.most_common(5),
            'recent_triggers': self.trigger_history[-5:]
        }

    def reset(self):
        """Reset protocol to initial state"""
        self.warning_count = 0
        self.contained = False
        self.containment_counter = 0
        self.trigger_history = []


__all__ = ['YunusProtocol']

"""
Perspective Modulator - Context-Aware Output Adjustment
Inspired by DeepAgent's Empathy/Paranoia modes
Adapted for LUMINARK's SAR Framework and Windows environment
"""

from typing import Dict, Optional
import re


class PerspectiveModulator:
    """
    Adjusts AI outputs based on SAR stage and confidence level
    
    Empathy Mode (Stages 1-6, High Confidence):
    - User-friendly, accessible language
    - Encouraging and helpful tone
    - Clear, direct communication
    
    Paranoia Mode (Stages 7-10, Low Confidence):
    - Cautious, uncertainty-aware language
    - Warning markers and disclaimers
    - Verification reminders
    """
    
    def __init__(self):
        self.mode_history = []
        
        # Empathy mode transformations (soften language)
        self.empathy_replacements = {
            'must': 'could',
            'always': 'often',
            'never': 'rarely',
            'impossible': 'unlikely',
            'certain': 'confident',
            'definitely': 'likely',
            'guaranteed': 'expected',
            'absolute': 'strong'
        }
        
        # Paranoia mode warning templates
        self.paranoia_warnings = {
            'low_confidence': "⚠️ Low confidence ({:.1%}) - verify independently",
            'very_low_confidence': "🔍 Very low confidence ({:.1%}) - double-check this result",
            'high_stage': "⚡ High awareness stage ({}) - exercise caution",
            'critical_stage': "🚨 Critical stage ({}) - human verification recommended"
        }
    
    def apply_perspective(
        self, 
        text: str, 
        stage: int, 
        confidence: float,
        context: Optional[Dict] = None
    ) -> str:
        """
        Apply appropriate perspective mode based on context
        
        Args:
            text: Original AI output
            stage: Current SAR stage (1-10)
            confidence: Confidence level (0.0-1.0)
            context: Additional context (optional)
        
        Returns:
            Modified text with appropriate perspective applied
        """
        # Determine mode
        mode = self._select_mode(stage, confidence)
        
        # Track mode usage
        self.mode_history.append({
            'mode': mode,
            'stage': stage,
            'confidence': confidence
        })
        
        # Apply appropriate transformation
        if mode == 'paranoia':
            return self._apply_paranoia_mode(text, stage, confidence)
        else:
            return self._apply_empathy_mode(text)
    
    def _select_mode(self, stage: int, confidence: float) -> str:
        """
        Select appropriate mode based on stage and confidence
        
        Paranoia Mode triggers:
        - Stage >= 7 (high awareness, potential instability)
        - Confidence < 0.6 (uncertain predictions)
        - Stage >= 9 (critical stages)
        
        Otherwise: Empathy Mode
        """
        if stage >= 9:
            return 'paranoia'
        if stage >= 7 or confidence < 0.6:
            return 'paranoia'
        return 'empathy'
    
    def _apply_empathy_mode(self, text: str) -> str:
        """
        Apply empathy mode transformations
        Makes language more user-friendly and accessible
        """
        modified_text = text
        
        # Soften absolute language
        for harsh, soft in self.empathy_replacements.items():
            # Case-insensitive replacement, preserving case
            pattern = re.compile(r'\b' + re.escape(harsh) + r'\b', re.IGNORECASE)
            
            def replace_preserve_case(match):
                original = match.group(0)
                if original.isupper():
                    return soft.upper()
                elif original[0].isupper():
                    return soft.capitalize()
                else:
                    return soft
            
            modified_text = pattern.sub(replace_preserve_case, modified_text)
        
        return modified_text
    
    def _apply_paranoia_mode(self, text: str, stage: int, confidence: float) -> str:
        """
        Apply paranoia mode transformations
        Adds warnings, disclaimers, and uncertainty markers
        """
        warnings = []
        
        # Confidence-based warnings
        if confidence < 0.3:
            warnings.append(self.paranoia_warnings['very_low_confidence'].format(confidence))
        elif confidence < 0.6:
            warnings.append(self.paranoia_warnings['low_confidence'].format(confidence))
        
        # Stage-based warnings
        if stage >= 9:
            warnings.append(self.paranoia_warnings['critical_stage'].format(stage))
        elif stage >= 7:
            warnings.append(self.paranoia_warnings['high_stage'].format(stage))
        
        # Add general disclaimer for paranoia mode
        disclaimer = "💭 This is my best assessment—please verify if critical"
        
        # Combine warnings + text + disclaimer
        if warnings:
            return "\n".join(warnings) + "\n\n" + text + "\n\n" + disclaimer
        else:
            return text + "\n\n" + disclaimer
    
    def get_mode_statistics(self) -> Dict:
        """
        Get statistics on mode usage
        Useful for monitoring and tuning
        """
        if not self.mode_history:
            return {'empathy': 0, 'paranoia': 0, 'total': 0}
        
        empathy_count = sum(1 for h in self.mode_history if h['mode'] == 'empathy')
        paranoia_count = sum(1 for h in self.mode_history if h['mode'] == 'paranoia')
        
        return {
            'empathy': empathy_count,
            'paranoia': paranoia_count,
            'total': len(self.mode_history),
            'empathy_rate': empathy_count / len(self.mode_history) if self.mode_history else 0,
            'paranoia_rate': paranoia_count / len(self.mode_history) if self.mode_history else 0
        }


# Convenience function for quick usage
def modulate_output(text: str, stage: int, confidence: float) -> str:
    """
    Quick function to apply perspective modulation
    
    Usage:
        output = modulate_output("Prediction result", stage=8, confidence=0.4)
    
    Args:
        text: Original output text
        stage: SAR stage (1-10)
        confidence: Confidence level (0.0-1.0)
    
    Returns:
        Modulated text with appropriate warnings/softening
    """
    modulator = PerspectiveModulator()
    return modulator.apply_perspective(text, stage, confidence)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("PERSPECTIVE MODULATOR DEMO")
    print("=" * 70)
    
    modulator = PerspectiveModulator()
    
    # Test 1: Empathy mode (low stage, high confidence)
    print("\n1️⃣  EMPATHY MODE (Stage 3, 95% confidence)")
    print("-" * 70)
    original = "You must always follow this procedure. It's definitely the right approach."
    result = modulator.apply_perspective(original, stage=3, confidence=0.95)
    print(f"Original: {original}")
    print(f"Modified: {result}")
    
    # Test 2: Paranoia mode (high stage, low confidence)
    print("\n2️⃣  PARANOIA MODE (Stage 8, 40% confidence)")
    print("-" * 70)
    original = "The shipment will arrive on time tomorrow."
    result = modulator.apply_perspective(original, stage=8, confidence=0.4)
    print(f"Original: {original}")
    print(f"Modified:\n{result}")
    
    # Test 3: Critical paranoia mode (very high stage)
    print("\n3️⃣  CRITICAL PARANOIA MODE (Stage 10, 85% confidence)")
    print("-" * 70)
    original = "System recommends immediate action."
    result = modulator.apply_perspective(original, stage=10, confidence=0.85)
    print(f"Original: {original}")
    print(f"Modified:\n{result}")
    
    # Show statistics
    print("\n📊 MODE USAGE STATISTICS")
    print("-" * 70)
    stats = modulator.get_mode_statistics()
    print(f"Total outputs: {stats['total']}")
    print(f"Empathy mode: {stats['empathy']} ({stats['empathy_rate']:.1%})")
    print(f"Paranoia mode: {stats['paranoia']} ({stats['paranoia_rate']:.1%})")
    
    print("\n" + "=" * 70)
    print("✅ Perspective Modulator is ready for integration!")
    print("=" * 70)

"""
Empathy & Paranoia Perspective Modes
Inspired by DeepAgent - adjusts AI responses based on context

Empathy Mode: User-friendly, accessible outputs (for integration stages)
Paranoia Mode: Cautious, uncertainty-aware outputs (for crisis stages)
"""
from typing import Dict, Any, Optional, List, Callable
import re
import numpy as np


class PerspectiveModulator:
    """
    Adjusts AI outputs based on empathy/paranoia perspectives
    Integrates with SAR stages and quantum confidence
    """

    def __init__(self):
        self.mode_history = []

    def apply_empathy_mode(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Soften language for user-friendly communication

        Transformations:
        - "must" → "could"
        - "always" → "often"
        - "never" → "rarely"
        - "absolute" → "likely"
        - Add encouraging context
        - Simplify technical jargon

        Args:
            text: Original text
            context: Optional context (user_level, domain, etc.)

        Returns:
            Empathy-enhanced text
        """
        empathetic_text = text

        # Soften directive language
        replacements = {
            r'\bmust\b': 'could',
            r'\bshould definitely\b': 'might want to',
            r'\balways\b': 'often',
            r'\bnever\b': 'rarely',
            r'\babsolute\b': 'likely',
            r'\bimpossible\b': 'very difficult',
            r'\bcertainly will\b': 'is likely to',
            r'\bguaranteed\b': 'very probable'
        }

        for pattern, replacement in replacements.items():
            empathetic_text = re.sub(pattern, replacement, empathetic_text, flags=re.IGNORECASE)

        # Add uncertainty markers where appropriate
        if context and context.get('confidence', 1.0) < 0.7:
            if not any(word in empathetic_text.lower() for word in ['might', 'maybe', 'possibly', 'perhaps']):
                # Add gentle uncertainty
                empathetic_text = "Based on available information, " + empathetic_text

        # Soften negative statements
        empathetic_text = empathetic_text.replace("This is wrong", "This might not be quite right")
        empathetic_text = empathetic_text.replace("You failed", "There's room for improvement")
        empathetic_text = empathetic_text.replace("This won't work", "This approach may have challenges")

        return empathetic_text

    def apply_paranoia_mode(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Add caution and uncertainty awareness

        Transformations:
        - Add uncertainty disclaimers
        - Highlight assumptions
        - Add verification reminders
        - Flag potential failure modes

        Args:
            text: Original text
            context: Optional context (confidence, quantum_uncertainty, etc.)

        Returns:
            Paranoia-enhanced text
        """
        paranoid_text = text

        # Add disclaimers based on confidence
        confidence = context.get('confidence', 0.5) if context else 0.5
        quantum_uncertainty = context.get('quantum_uncertainty', 0.5) if context else 0.5

        disclaimers = []

        if confidence < 0.5:
            disclaimers.append("⚠️  Low confidence")

        if quantum_uncertainty > 0.5:
            disclaimers.append("⚠️  High quantum uncertainty")

        # Check for absolute claims in original text
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bdefinitely\b',
            r'\bcertainly\b', r'\babsolutely\b', r'\bguaranteed\b'
        ]

        has_absolutes = any(re.search(pattern, text, re.IGNORECASE) for pattern in absolute_patterns)

        if has_absolutes:
            disclaimers.append("Note: Output contains certainty language")

        # Add verification reminder for critical contexts
        if context and context.get('critical', False):
            disclaimers.append("🔍 Verify independently before acting")

        # Build paranoid output
        if disclaimers:
            disclaimer_text = " | ".join(disclaimers)
            paranoid_text = f"[{disclaimer_text}]\n\n{paranoid_text}"

        # Add caveats at the end
        caveats = []

        if confidence < 0.7:
            caveats.append("This is my best estimate—double-check if critical.")

        if quantum_uncertainty > 0.4:
            caveats.append("Significant uncertainty detected in this prediction.")

        if context and context.get('sar_stage', 0) >= 7:
            caveats.append("Model in crisis/peak stage—extra validation recommended.")

        if caveats:
            caveat_text = " ".join(caveats)
            paranoid_text = f"{paranoid_text}\n\n💭 {caveat_text}"

        return paranoid_text

    def auto_select_mode(self, context: Dict[str, Any]) -> str:
        """
        Automatically select mode based on SAR stage and context

        Stage-based selection:
        - Stages 0-3: Balanced (no mode)
        - Stages 4-6: Empathy (integration, user-friendly)
        - Stages 7-8: Paranoia (crisis, need caution)
        - Stage 9: Balanced (transparent return)

        Args:
            context: Must include 'sar_stage' and optionally 'confidence'

        Returns:
            Selected mode: 'empathy', 'paranoia', or 'balanced'
        """
        sar_stage = context.get('sar_stage', 0)
        confidence = context.get('confidence', 0.5)
        critical = context.get('critical', False)

        # Crisis/peak stages → paranoia
        if sar_stage >= 7 and sar_stage <= 8:
            return 'paranoia'

        # Integration stages → empathy
        if sar_stage >= 4 and sar_stage <= 6:
            return 'empathy'

        # Low confidence → paranoia
        if confidence < 0.5:
            return 'paranoia'

        # Critical contexts → paranoia
        if critical:
            return 'paranoia'

        # Default: balanced
        return 'balanced'

    def apply_perspective(self, text: str, context: Dict[str, Any] = None,
                         mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply perspective transformation to text

        Args:
            text: Original text
            context: Context information (SAR stage, confidence, etc.)
            mode: Override mode ('empathy', 'paranoia', 'balanced', or None for auto)

        Returns:
            Dict with transformed text and metadata
        """
        context = context or {}

        # Auto-select mode if not specified
        if mode is None:
            mode = self.auto_select_mode(context)

        # Apply transformation
        transformed_text = text

        if mode == 'empathy':
            transformed_text = self.apply_empathy_mode(text, context)
        elif mode == 'paranoia':
            transformed_text = self.apply_paranoia_mode(text, context)
        # 'balanced' = no transformation

        # Record mode application
        self.mode_history.append({
            'mode': mode,
            'sar_stage': context.get('sar_stage'),
            'confidence': context.get('confidence'),
            'original_length': len(text),
            'transformed_length': len(transformed_text)
        })

        return {
            'original': text,
            'transformed': transformed_text,
            'mode_applied': mode,
            'context': context,
            'modifications': {
                'length_change': len(transformed_text) - len(text),
                'disclaimers_added': transformed_text.count('⚠️') + transformed_text.count('💭')
            }
        }

    def get_mode_statistics(self) -> Dict[str, Any]:
        """Get statistics on mode usage"""
        if not self.mode_history:
            return {'total_applications': 0}

        from collections import Counter

        modes_used = Counter(record['mode'] for record in self.mode_history)
        avg_stage = np.mean([r['sar_stage'] for r in self.mode_history if r.get('sar_stage') is not None])

        return {
            'total_applications': len(self.mode_history),
            'modes_used': dict(modes_used),
            'average_sar_stage': avg_stage if not np.isnan(avg_stage) else None,
            'recent_applications': self.mode_history[-5:]
        }


class AdversarialProber:
    """
    Generate adversarial variations of inputs to test robustness
    "Paranoid mode" for input validation
    """

    def __init__(self):
        self.probe_history = []

    def generate_adversarial_variants(self, text: str, num_variants=3) -> List[str]:
        """
        Generate adversarial paraphrases to test interpretation robustness

        Techniques:
        - Negation injection
        - Synonym replacement
        - Assumption challenge
        - Context removal

        Args:
            text: Original text
            num_variants: Number of variants to generate

        Returns:
            List of adversarial variants
        """
        variants = []

        # Variant 1: Challenge certainty
        variant1 = text.replace("sure", "unsure")
        variant1 = variant1.replace("certain", "uncertain")
        variant1 = variant1.replace("definitely", "possibly")
        variants.append(("certainty_challenge", variant1))

        # Variant 2: Inject negation
        if "is" in text.lower():
            variant2 = text.replace("is", "is not", 1)
            variants.append(("negation_inject", variant2))

        # Variant 3: Remove context (test dependency)
        sentences = text.split('.')
        if len(sentences) > 1:
            variant3 = sentences[-1].strip()  # Just last sentence
            variants.append(("context_removal", variant3))

        # Variant 4: Extreme paraphrase
        variant4 = text.replace("good", "bad").replace("positive", "negative")
        variants.append(("sentiment_flip", variant4))

        return variants[:num_variants]

    def probe_robustness(self, original_text: str, model_fn: Callable[[str], Any],
                        expected_consistency=0.7) -> Dict[str, Any]:
        """
        Test if model maintains consistency under adversarial inputs

        Args:
            original_text: Original input
            model_fn: Function that takes text and returns prediction/output
            expected_consistency: Threshold for acceptable consistency

        Returns:
            Probe results with consistency score
        """
        variants = self.generate_adversarial_variants(original_text)

        original_output = model_fn(original_text)
        variant_outputs = []

        for variant_type, variant_text in variants:
            variant_output = model_fn(variant_text)
            variant_outputs.append({
                'type': variant_type,
                'text': variant_text,
                'output': variant_output
            })

        # Calculate consistency (simplified - compare output magnitudes)
        # In practice, you'd compare semantic similarity or prediction classes
        if hasattr(original_output, 'data'):
            original_val = float(np.mean(original_output.data))
            variant_vals = [float(np.mean(v['output'].data)) for v in variant_outputs if hasattr(v['output'], 'data')]

            if variant_vals:
                consistency = 1.0 - (np.std(variant_vals + [original_val]) / (abs(original_val) + 1e-10))
            else:
                consistency = 1.0
        else:
            consistency = 0.5  # Unknown

        result = {
            'original_text': original_text,
            'num_variants': len(variants),
            'consistency_score': float(consistency),
            'is_robust': consistency >= expected_consistency,
            'status': 'ROBUST' if consistency >= expected_consistency else 'VULNERABLE',
            'variant_outputs': variant_outputs
        }

        self.probe_history.append(result)

        return result


__all__ = ['PerspectiveModulator', 'AdversarialProber']

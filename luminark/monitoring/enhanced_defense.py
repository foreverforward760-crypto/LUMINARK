"""
Enhanced Multi-Stage Awareness Defense System
Combines mycelial defense with 10-stage awareness detection
"""
import numpy as np
from typing import Dict, List, Optional
from enum import Enum


class AwarenessStage(Enum):
    """10-stage awareness system inspired by SAR framework"""
    STAGE_0_RECEPTIVE = 0  # Plenara - Open, receptive, questioning
    STAGE_1_FOUNDATION = 1  # Building understanding
    STAGE_2_EXPLORATION = 2  # Active learning
    STAGE_3_INTEGRATION = 3  # Connecting concepts
    STAGE_4_EQUILIBRIUM = 4  # Balanced understanding
    STAGE_5_THRESHOLD = 5  # At edge of capabilities
    STAGE_6_EXPANSION = 6  # Pushing boundaries
    STAGE_7_WARNING = 7  # Potential overconfidence/hallucination
    STAGE_8_CRITICAL = 8  # Dangerous overreach (omniscience trap)
    STAGE_9_RENEWAL = 9  # Self-aware humility, ready to restart


class EnhancedDefenseSystem:
    """
    Multi-layered defense with awareness stage detection
    Protects against overconfidence, hallucination, and training instability
    """

    def __init__(self):
        # Stage detection thresholds
        self.stage_thresholds = {
            'stability': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.95],
            'confidence': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.5],
            'coherence': [0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.9]
        }

        # Defense protocols by stage
        self.stage_protocols = {
            AwarenessStage.STAGE_0_RECEPTIVE: {
                'name': 'Receptive Learning',
                'actions': ['encourage_exploration', 'accept_uncertainty'],
                'risk_level': 'low',
                'description': 'Open receptive state, ready to learn'
            },
            AwarenessStage.STAGE_4_EQUILIBRIUM: {
                'name': 'Balanced Operation',
                'actions': ['maintain_balance', 'steady_learning'],
                'risk_level': 'nominal',
                'description': 'Healthy balanced learning state'
            },
            AwarenessStage.STAGE_5_THRESHOLD: {
                'name': 'Threshold Warning',
                'actions': ['increase_monitoring', 'caution_advised'],
                'risk_level': 'elevated',
                'description': 'Approaching limits of reliable knowledge'
            },
            AwarenessStage.STAGE_7_WARNING: {
                'name': 'Hallucination Risk',
                'actions': ['reduce_confidence', 'verify_claims', 'request_evidence'],
                'risk_level': 'high',
                'description': 'High risk of overconfident predictions or hallucinations'
            },
            AwarenessStage.STAGE_8_CRITICAL: {
                'name': 'Omniscience Trap',
                'actions': ['emergency_stop', 'require_human_review', 'reset_confidence'],
                'risk_level': 'critical',
                'description': 'Dangerous overreach - model thinks it knows more than it does'
            },
            AwarenessStage.STAGE_9_RENEWAL: {
                'name': 'Humble Restart',
                'actions': ['acknowledge_limits', 'restart_learning', 'transparency'],
                'risk_level': 'recovery',
                'description': 'Self-aware of limitations, ready to learn properly'
            }
        }

        # Alert history
        self.alert_history = []
        self.max_history = 100

    def analyze_training_state(self, metrics: Dict) -> Dict:
        """
        Analyze training state and determine awareness stage

        Args:
            metrics: Dict containing:
                - loss: Current loss value
                - accuracy: Current accuracy
                - grad_norm: Gradient norm
                - loss_variance: Variance in recent losses
                - confidence: Model confidence score

        Returns:
            Analysis dict with stage, risk level, and recommended actions
        """
        # Extract metrics
        loss = metrics.get('loss', 1.0)
        accuracy = metrics.get('accuracy', 0.5)
        grad_norm = metrics.get('grad_norm', 1.0)
        loss_variance = metrics.get('loss_variance', 0.1)
        confidence = metrics.get('confidence', 0.7)

        # Calculate awareness indicators
        stability = self._calculate_stability(loss, grad_norm, loss_variance)
        coherence = self._calculate_coherence(accuracy, loss, confidence)

        # Determine awareness stage
        stage = self._determine_stage(stability, confidence, coherence)

        # Get protocol for this stage
        protocol = self.stage_protocols.get(stage, self.stage_protocols[AwarenessStage.STAGE_4_EQUILIBRIUM])

        # Determine if defensive action needed
        defense_activated = stage.value >= 5  # Threshold or above

        # Create analysis result
        analysis = {
            'stage': stage,
            'stage_name': stage.name,
            'stage_value': stage.value,
            'protocol': protocol,
            'defense_activated': defense_activated,
            'metrics': {
                'stability': stability,
                'confidence': confidence,
                'coherence': coherence,
                'loss': loss,
                'accuracy': accuracy
            },
            'recommended_actions': protocol['actions'],
            'risk_level': protocol['risk_level'],
            'description': protocol['description']
        }

        # Log alert if risk elevated
        if protocol['risk_level'] in ['elevated', 'high', 'critical']:
            self._log_alert(analysis)

        return analysis

    def _calculate_stability(self, loss: float, grad_norm: float, loss_variance: float) -> float:
        """
        Calculate training stability
        High stability = low loss, reasonable gradients, low variance
        """
        # Normalize loss (assuming typical range 0-5)
        loss_stability = max(0, 1 - loss / 5.0)

        # Normalize gradient (assuming typical range 0-10)
        grad_stability = max(0, 1 - grad_norm / 10.0)

        # Normalize variance (assuming typical range 0-1)
        variance_stability = max(0, 1 - loss_variance)

        # Weighted combination
        stability = 0.4 * loss_stability + 0.3 * grad_stability + 0.3 * variance_stability
        return float(np.clip(stability, 0, 1))

    def _calculate_coherence(self, accuracy: float, loss: float, confidence: float) -> float:
        """
        Calculate prediction coherence
        High coherence = accuracy matches confidence, loss is reasonable
        """
        # Check if confidence matches accuracy (calibration)
        calibration = 1 - abs(confidence - accuracy)

        # Check if loss is reasonable for the accuracy
        expected_loss = -np.log(accuracy + 0.01)  # Rough estimate
        loss_coherence = max(0, 1 - abs(loss - expected_loss) / expected_loss)

        # Weighted combination
        coherence = 0.6 * calibration + 0.4 * loss_coherence
        return float(np.clip(coherence, 0, 1))

    def _determine_stage(self, stability: float, confidence: float, coherence: float) -> AwarenessStage:
        """Determine awareness stage based on metrics"""

        # Stage 8 (Critical): Very high confidence with low stability/coherence
        if confidence > 0.95 and (stability < 0.2 or coherence < 0.3):
            return AwarenessStage.STAGE_8_CRITICAL

        # Stage 7 (Warning): High confidence with moderate issues
        if confidence > 0.9 and (stability < 0.4 or coherence < 0.5):
            return AwarenessStage.STAGE_7_WARNING

        # Stage 5 (Threshold): Moderate confidence, approaching limits
        if confidence > 0.8 and stability < 0.5:
            return AwarenessStage.STAGE_5_THRESHOLD

        # Stage 9 (Renewal): Low confidence but high stability (learned humility)
        if confidence < 0.5 and stability > 0.9 and coherence > 0.8:
            return AwarenessStage.STAGE_9_RENEWAL

        # Stage 0 (Receptive): Low confidence, moderate stability (early learning)
        if confidence < 0.4 and stability > 0.7:
            return AwarenessStage.STAGE_0_RECEPTIVE

        # Stage 4 (Equilibrium): Balanced state
        if 0.6 < confidence < 0.8 and stability > 0.6 and coherence > 0.6:
            return AwarenessStage.STAGE_4_EQUILIBRIUM

        # Stage 3 (Integration): Building up
        if 0.5 < confidence < 0.7 and stability > 0.5:
            return AwarenessStage.STAGE_3_INTEGRATION

        # Stage 6 (Expansion): Pushing boundaries
        if confidence > 0.75 and stability > 0.6:
            return AwarenessStage.STAGE_6_EXPANSION

        # Default to Stage 2 (Exploration)
        return AwarenessStage.STAGE_2_EXPLORATION

    def _log_alert(self, analysis: Dict):
        """Log a defense alert"""
        alert = {
            'stage': analysis['stage_name'],
            'risk_level': analysis['risk_level'],
            'actions': analysis['recommended_actions'],
            'metrics': analysis['metrics']
        }
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts"""
        if not self.alert_history:
            return {'total_alerts': 0, 'by_risk_level': {}}

        by_risk = {}
        for alert in self.alert_history:
            level = alert['risk_level']
            by_risk[level] = by_risk.get(level, 0) + 1

        return {
            'total_alerts': len(self.alert_history),
            'by_risk_level': by_risk,
            'most_recent': self.alert_history[-1] if self.alert_history else None
        }

    def should_stop_training(self, analysis: Dict) -> bool:
        """Determine if training should be stopped"""
        # Stop if critical stage reached
        if analysis['stage'] == AwarenessStage.STAGE_8_CRITICAL:
            return True

        # Stop if too many high-risk alerts recently
        recent_alerts = self.alert_history[-10:]
        high_risk_count = sum(1 for alert in recent_alerts
                             if alert['risk_level'] in ['high', 'critical'])
        if high_risk_count >= 5:
            return True

        return False

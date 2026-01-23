#!/usr/bin/env python3
"""
Standalone SAP Diagnostic Test
Complete self-contained implementation
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class SAPStage(Enum):
    """SAP developmental stages"""
    STAGE_0_PLENARA = 0
    STAGE_1_PULSE = 1
    STAGE_2_DUALITY = 2
    STAGE_3_STABLE_FORM = 3
    STAGE_4_FOUNDATION = 4
    STAGE_5_BILATERAL_THRESHOLD = 5
    STAGE_6_INTEGRATION = 6
    STAGE_7_CRISIS = 7
    STAGE_8_UNITY_PEAK = 8
    STAGE_9_TRANSPARENT_RETURN = 9


class TrajectoryType(Enum):
    """Predicted developmental trajectory"""
    PROGRESSIVE = "progressive"
    REGRESSIVE = "regressive"
    STAGNANT = "stagnant"
    UNCERTAIN = "uncertain"


@dataclass
class SAPAssessment:
    """Complete SAP stage assessment result"""
    stage: int
    stage_description: str
    confidence: float
    ambiguity_margin: float
    second_best_stage: Optional[int]
    insights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trajectory_type: str = "UNCERTAIN"
    trajectory_predictions: List[str] = field(default_factory=list)
    next_stage: int = 0
    input_scores: Dict[str, float] = field(default_factory=dict)
    stage_matches: Optional[Dict[int, float]] = None


class SAPDiagnostic:
    """SAP Stage Diagnostic Tool - Empirically Calibrated"""

    __version__ = "4.3.0"

    STAGE_DESCRIPTIONS = {
        0: "Plenara / Void: Undifferentiated potential. No active structure.",
        1: "Pulse / Flash: Initial impulse. Energy without clear direction.",
        2: "Duality: First binary split. Basic opposition or choice emerges.",
        3: "Stable Form: First coherent structure. Functional triad forms.",
        4: "Foundation: System building. Establishing stable routines and structures.",
        5: "Bilateral Threshold: Critical decision point. Maximum tension and potential for progress/regression.",
        6: "Integration: Harmonious complexity. Multiple elements working together effectively.",
        7: "Crisis / Purification: High-stress reorganization. Old structures break down for higher integration.",
        8: "Unity Peak: Coherent, high-functioning whole. Risk of permanence trap.",
        9: "Transparent Return: Cycle completion. Awareness of the whole process, renewal ready."
    }

    CRITERIA = {
        'complexity': "Degree of structural complexity and interconnection",
        'stability': "System resilience and predictability",
        'tension': "Level of internal stress or conflict",
        'adaptability': "Capacity to change in response to challenges",
        'coherence': "Sense of unified purpose or function"
    }

    def __init__(self):
        """Initialize with empirically-refined stage weights"""
        self.stage_weights = {
            0: {'complexity': 0.0, 'stability': 0.0, 'tension': 0.0, 'adaptability': 0.10, 'coherence': 0.0},
            1: {'complexity': 0.1, 'stability': 0.0, 'tension': 0.3, 'adaptability': 0.30, 'coherence': 0.0},
            2: {'complexity': 0.2, 'stability': 0.1, 'tension': 0.4, 'adaptability': 0.20, 'coherence': 0.1},
            3: {'complexity': 0.3, 'stability': 0.4, 'tension': 0.2, 'adaptability': 0.20, 'coherence': 0.3},
            4: {'complexity': 0.4, 'stability': 0.7, 'tension': 0.1, 'adaptability': 0.30, 'coherence': 0.5},
            5: {'complexity': 0.5, 'stability': 0.3, 'tension': 0.9, 'adaptability': 0.60, 'coherence': 0.4},
            6: {'complexity': 0.7, 'stability': 0.6, 'tension': 0.2, 'adaptability': 0.50, 'coherence': 0.8},
            7: {'complexity': 0.6, 'stability': 0.2, 'tension': 0.8, 'adaptability': 0.85, 'coherence': 0.3},
            8: {'complexity': 0.8, 'stability': 0.8, 'tension': 0.1, 'adaptability': 0.80, 'coherence': 0.9},
            9: {'complexity': 0.9, 'stability': 0.5, 'tension': 0.3, 'adaptability': 0.95, 'coherence': 0.7}
        }

    def _calculate_stage_matches(self, scores: Dict[str, float]) -> Dict[int, float]:
        """Calculate match score for each stage"""
        stage_matches = {}
        for stage in range(10):
            match_score = 0
            total_weight = 0
            for criterion, weight in self.stage_weights[stage].items():
                expected_value = weight * 10
                user_value = scores[criterion]
                difference = abs(expected_value - user_value)
                criterion_match = max(0, (10 - difference) / 10)
                match_score += criterion_match * weight
                total_weight += weight
            stage_matches[stage] = match_score / total_weight if total_weight > 0 else 0
        return stage_matches

    def assess(self, scores: Dict[str, float], verbose: bool = True) -> SAPAssessment:
        """Assess SAP stage from input scores"""
        stage_matches = self._calculate_stage_matches(scores)
        best_stage = max(stage_matches, key=stage_matches.get)
        confidence = stage_matches[best_stage]
        sorted_stages = sorted(stage_matches.items(), key=lambda x: x[1], reverse=True)
        second_best = sorted_stages[1] if len(sorted_stages) > 1 else (None, 0)
        ambiguity = confidence - second_best[1] if second_best[0] is not None else 1.0

        assessment = SAPAssessment(
            stage=best_stage,
            stage_description=self.STAGE_DESCRIPTIONS[best_stage],
            confidence=round(confidence, 3),
            ambiguity_margin=round(ambiguity, 3),
            second_best_stage=second_best[0],
            next_stage=(best_stage + 1) % 10 if best_stage < 9 else 0,
            input_scores=scores
        )

        if verbose:
            assessment.stage_matches = {s: round(v, 3) for s, v in stage_matches.items()}

        # Stage 9 authenticity check
        if best_stage == 9:
            authentic = (
                scores['complexity'] >= 8 and
                5 <= scores['stability'] <= 7 and
                2 <= scores['tension'] <= 5 and
                scores['adaptability'] >= 8 and
                scores['coherence'] >= 7
            )
            if authentic:
                assessment.insights.append("AUTHENTIC STAGE 9: Pattern consistent with transparent return. Genuine cycle completion.")
            else:
                assessment.warnings.append("STAGE 9 QUESTIONABLE: Pattern doesn't fully match transparent return criteria.")

        # Trajectory prediction
        if scores['adaptability'] >= 7:
            assessment.trajectory_type = "PROGRESSIVE"
        elif scores['adaptability'] <= 3:
            assessment.trajectory_type = "REGRESSIVE/TRAP" if best_stage >= 6 else "STAGNANT"
        else:
            assessment.trajectory_type = "UNCERTAIN"

        return assessment

    def generate_report(self, scores: Dict[str, float]) -> str:
        """Generate human-readable assessment report"""
        result = self.assess(scores, verbose=True)

        report = [
            "=" * 80,
            f"SAP STAGE DIAGNOSTIC v{self.__version__} – ASSESSMENT REPORT",
            "Stanfield's Axiom of Perpetuity (SAP) - Developmental Stage Analysis",
            "=" * 80,
            "",
            f"PRIMARY STAGE: {result.stage}",
            f"{result.stage_description}",
            f"Confidence: {result.confidence * 100:.1f}%",
            f"Ambiguity Margin: {result.ambiguity_margin:.3f}",
        ]

        if result.second_best_stage is not None:
            report.append(f"Second-Best: Stage {result.second_best_stage} ({result.stage_matches[result.second_best_stage] * 100:.1f}%)")
        report.append("")

        report.append("INPUT SCORES:")
        for criterion, score in scores.items():
            report.append(f"  {criterion.capitalize()}: {score}/10")
        report.append("")

        if result.warnings:
            report.append("⚠️  WARNINGS:")
            for w in result.warnings:
                report.append(f"  • {w}")
            report.append("")

        if result.insights:
            report.append("💡 INSIGHTS:")
            for i in result.insights:
                report.append(f"  • {i}")
            report.append("")

        report.append(f"🔮 TRAJECTORY: {result.trajectory_type}")
        report.append("")

        report.append("STAGE DISTRIBUTION:")
        for stage in range(10):
            pct = result.stage_matches[stage] * 100
            bar = "█" * int(pct / 5)
            report.append(f"  Stage {stage}: {bar} {pct:.1f}%")
        report.append("")

        report.extend([
            "=" * 80,
            f"SAP Diagnostic v{self.__version__} | Stanfield's Axiom of Perpetuity",
            "Project: LUMINARK - AI Safety Research",
            "Copyright © 2024-2025 Richard Leroy Stanfield Jr. All rights reserved.",
            "=" * 80
        ])

        return "\n".join(report)


def main():
    print("Testing SAP Diagnostic with provided scores...")
    print()

    diagnostic = SAPDiagnostic()

    scores = {
        'complexity': 8.0,
        'stability': 5.0,
        'tension': 3.0,
        'adaptability': 9.0,
        'coherence': 7.0
    }

    report = diagnostic.generate_report(scores)
    print(report)


if __name__ == "__main__":
    main()

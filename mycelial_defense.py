#!/usr/bin/env python3
"""
LUMINARK Mycelial Defense System
Advanced threat detection and response mechanism
"""


class MycelialDefenseSystem:
    """
    Mycelial Defense System monitors system metrics and triggers
    appropriate defense strategies based on threat conditions.
    """

    # Defense mode thresholds
    THRESHOLDS = {
        'octo_camouflage': {
            'tension': 0.80,
            'coherence_max': 0.30,
        },
        'mycelial_wrap': {
            'stability_max': 0.20,
            'tension': 0.70,
        },
        'harrowing': {
            'stability_max': 0.10,
            'tension': 0.90,
            'coherence_max': 0.20,
        }
    }

    def __init__(self):
        self.defense_mode = "NOMINAL"
        self.strategy = "Standard monitoring"
        self.alert_level = 0

    def analyze_threat(self, stability=1.0, tension=0.0, coherence=1.0):
        """
        Analyze system metrics and determine appropriate defense response.

        Args:
            stability (float): System stability metric (0.0 - 1.0)
            tension (float): System tension/stress metric (0.0 - 1.0)
            coherence (float): System coherence metric (0.0 - 1.0)

        Returns:
            dict: Defense analysis with mode, strategy, and metrics
        """

        # Check for HARROWING condition (most severe)
        if (stability <= self.THRESHOLDS['harrowing']['stability_max'] and
            tension >= self.THRESHOLDS['harrowing']['tension'] and
            coherence <= self.THRESHOLDS['harrowing']['coherence_max']):

            self.defense_mode = "HARROWING"
            self.strategy = "Full mycelial network lockdown - emergency protocols active"
            self.alert_level = 3

        # Check for MYCELIAL WRAP condition
        elif (stability <= self.THRESHOLDS['mycelial_wrap']['stability_max'] and
              tension >= self.THRESHOLDS['mycelial_wrap']['tension']):

            self.defense_mode = "MYCELIAL_WRAP"
            self.strategy = "Defensive encapsulation - isolate and contain threats"
            self.alert_level = 2

        # Check for OCTO-CAMOUFLAGE condition
        elif (tension >= self.THRESHOLDS['octo_camouflage']['tension'] and
              coherence <= self.THRESHOLDS['octo_camouflage']['coherence_max']):

            self.defense_mode = "OCTO_CAMOUFLAGE"
            self.strategy = "Adaptive stealth mode - blend and evade detection"
            self.alert_level = 1

        # No threats detected
        else:
            self.defense_mode = "NOMINAL"
            self.strategy = "Standard monitoring - all systems nominal"
            self.alert_level = 0

        return {
            'defense_mode': self.defense_mode,
            'strategy': self.strategy,
            'alert_level': self.alert_level,
            'metrics': {
                'stability': stability,
                'tension': tension,
                'coherence': coherence
            }
        }

    def get_defense_description(self):
        """Get detailed description of current defense mode"""
        descriptions = {
            "NOMINAL": {
                "status": "All systems operating normally",
                "actions": ["Continuous monitoring", "Standard protocols active"],
                "threat_level": "None"
            },
            "OCTO_CAMOUFLAGE": {
                "status": "High tension detected - activating camouflage",
                "actions": [
                    "Obscure system signatures",
                    "Randomize communication patterns",
                    "Deploy decoy processes",
                    "Minimize observable footprint"
                ],
                "threat_level": "Moderate"
            },
            "MYCELIAL_WRAP": {
                "status": "Instability detected - deploying defensive wrap",
                "actions": [
                    "Isolate compromised segments",
                    "Strengthen network boundaries",
                    "Activate redundant pathways",
                    "Contain potential threats"
                ],
                "threat_level": "High"
            },
            "HARROWING": {
                "status": "CRITICAL - Full defensive harrowing activated",
                "actions": [
                    "Complete network lockdown",
                    "Emergency protocol execution",
                    "Deep threat analysis and purge",
                    "Rebuild core integrity",
                    "Establish safe perimeter"
                ],
                "threat_level": "Critical"
            }
        }

        return descriptions.get(self.defense_mode, descriptions["NOMINAL"])

    def __repr__(self):
        return f"MycelialDefenseSystem(mode={self.defense_mode}, alert_level={self.alert_level})"

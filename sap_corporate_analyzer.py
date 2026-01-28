"""
LUMINARK SAP CORPORATE ANALYZER
Economic/Financial Systems Diagnostic Tool

Maps corporations, markets, and economic entities to the 10-stage SAP framework
for investment intelligence and strategic analysis.
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SAPStage(Enum):
    """SAP Stage enumeration for corporate entities"""
    PLENARA = 0      # Pre-formation
    PULSE = 1        # First movement
    POLARITY = 2     # Binary split
    EXPRESSION = 3   # Breakthrough
    FOUNDATION = 4   # Equilibrium
    THRESHOLD = 5    # Crisis point
    INTEGRATION = 6  # Peak performance
    ANALYSIS = 7     # Purification crisis
    UNITY_PEAK = 8   # Permanence trap
    RELEASE = 9      # Transparent return


@dataclass
class SPATVectors:
    """5-Dimensional SPAT vectors for corporate analysis"""
    complexity: float      # 0-10: Information density, relationships, variables
    stability: float       # 0-10: Structural integrity, predictability
    tension: float         # 0-10: Internal pressure, drive for change
    adaptability: float    # 0-10: Capacity to pivot, learn, evolve
    coherence: float       # 0-10: Strategic alignment, cultural health


@dataclass
class CorporateMetrics:
    """Input metrics for corporate SAP analysis"""
    # Financial
    revenue_growth_yoy: float          # Year-over-year %
    profit_margin: float               # Operating margin %
    debt_to_equity: float             # Leverage ratio
    cash_reserves_months: float        # Months of runway
    r_d_spending_pct: float           # % of revenue
    
    # Organizational
    employee_turnover_pct: float       # Annual %
    leadership_tenure_years: float     # Average years
    management_layers: int             # Hierarchy depth
    employee_satisfaction: float       # 0-10 scale
    
    # Market
    market_share_pct: float           # % of total market
    market_growth_yoy: float          # Industry growth %
    competitive_threats: int           # Number of disruptors
    
    # Innovation
    new_products_last_2yrs: int       # Product launches
    innovation_index: float            # 0-10 composite score
    
    # Cultural
    ceo_tenure_years: float           # Years in role
    mission_clarity: float             # 0-10 employee understanding
    
    # External
    news_sentiment: float              # -1 to 1 scale
    analyst_rating: str                # "buy", "hold", "sell"


class CorporateSAPAnalyzer:
    """
    Main analyzer for mapping corporations to SAP stages
    """
    
    def __init__(self):
        self.stage_weights = self._initialize_stage_weights()
    
    def _initialize_stage_weights(self) -> Dict[int, SPATVectors]:
        """
        Define ideal SPAT vectors for each SAP stage in corporate context
        """
        return {
            0: SPATVectors(0.5, 0.5, 5.0, 9.5, 1.0),   # Plenara: Pre-formation
            1: SPATVectors(2.5, 2.0, 8.0, 8.5, 4.0),   # Pulse: First movement
            2: SPATVectors(3.5, 5.0, 6.0, 6.5, 6.0),   # Polarity: Binary split
            3: SPATVectors(5.0, 4.0, 7.0, 7.5, 7.0),   # Expression: Breakthrough
            4: SPATVectors(6.0, 8.5, 3.0, 5.5, 8.5),   # Foundation: Equilibrium
            5: SPATVectors(7.0, 4.0, 9.5, 7.0, 6.0),   # Threshold: Crisis point
            6: SPATVectors(8.0, 8.0, 3.0, 7.5, 9.5),   # Integration: Peak
            7: SPATVectors(7.0, 2.5, 9.0, 5.0, 3.5),   # Analysis: Crisis
            8: SPATVectors(9.5, 9.5, 2.0, 4.0, 8.0),   # Unity Peak: Trap
            9: SPATVectors(8.0, 7.0, 4.0, 9.8, 9.5),   # Release: Wisdom
        }
    
    def calculate_spat_vectors(self, metrics: CorporateMetrics) -> SPATVectors:
        """
        Convert corporate metrics into SPAT vectors
        """
        # COMPLEXITY: Employee count, product lines, markets, revenue streams
        complexity = self._calculate_complexity(metrics)
        
        # STABILITY: Revenue volatility, cash reserves, debt, turnover
        stability = self._calculate_stability(metrics)
        
        # TENSION: Market pressure, competitive threats, growth gaps
        tension = self._calculate_tension(metrics)
        
        # ADAPTABILITY: Innovation, decision speed, pivot history
        adaptability = self._calculate_adaptability(metrics)
        
        # COHERENCE: Mission clarity, strategic alignment, cultural health
        coherence = self._calculate_coherence(metrics)
        
        return SPATVectors(complexity, stability, tension, adaptability, coherence)
    
    def _calculate_complexity(self, m: CorporateMetrics) -> float:
        """Calculate complexity score (0-10)"""
        # Higher revenue growth = more complexity
        # More management layers = more complexity
        # More competitive threats = more complexity
        
        growth_factor = min(m.revenue_growth_yoy / 50, 1.0) * 3  # 0-3
        hierarchy_factor = min(m.management_layers / 10, 1.0) * 3  # 0-3
        market_factor = min(m.competitive_threats / 10, 1.0) * 4  # 0-4
        
        return min(growth_factor + hierarchy_factor + market_factor, 10.0)
    
    def _calculate_stability(self, m: CorporateMetrics) -> float:
        """Calculate stability score (0-10)"""
        # High cash reserves = more stability
        # Low debt = more stability
        # Low turnover = more stability
        # Positive profit margin = more stability
        
        cash_factor = min(m.cash_reserves_months / 24, 1.0) * 3  # 0-3
        debt_factor = max(0, (1 - m.debt_to_equity / 2)) * 2  # 0-2
        turnover_factor = max(0, (1 - m.employee_turnover_pct / 50)) * 3  # 0-3
        margin_factor = min(m.profit_margin / 30, 1.0) * 2  # 0-2
        
        return min(cash_factor + debt_factor + turnover_factor + margin_factor, 10.0)
    
    def _calculate_tension(self, m: CorporateMetrics) -> float:
        """Calculate tension score (0-10)"""
        # High competitive threats = more tension
        # Negative profit margin = more tension
        # High debt = more tension
        # Low employee satisfaction = more tension
        
        competition_factor = min(m.competitive_threats / 5, 1.0) * 3  # 0-3
        margin_factor = max(0, -m.profit_margin / 20) * 2  # 0-2 (negative margins)
        debt_factor = min(m.debt_to_equity / 2, 1.0) * 3  # 0-3
        satisfaction_factor = max(0, (1 - m.employee_satisfaction / 10)) * 2  # 0-2
        
        return min(competition_factor + margin_factor + debt_factor + satisfaction_factor, 10.0)
    
    def _calculate_adaptability(self, m: CorporateMetrics) -> float:
        """Calculate adaptability score (0-10)"""
        # High R&D spending = more adaptability
        # Recent product launches = more adaptability
        # High innovation index = more adaptability
        # Low CEO tenure (not too low) = more adaptability
        
        rd_factor = min(m.r_d_spending_pct / 20, 1.0) * 3  # 0-3
        product_factor = min(m.new_products_last_2yrs / 3, 1.0) * 2  # 0-2
        innovation_factor = m.innovation_index / 10 * 3  # 0-3
        
        # CEO tenure sweet spot: 3-10 years
        if m.ceo_tenure_years < 2:
            tenure_factor = 1.0
        elif 2 <= m.ceo_tenure_years <= 10:
            tenure_factor = 2.0
        else:
            tenure_factor = max(0, 2 - (m.ceo_tenure_years - 10) / 10)
        
        return min(rd_factor + product_factor + innovation_factor + tenure_factor, 10.0)
    
    def _calculate_coherence(self, m: CorporateMetrics) -> float:
        """Calculate coherence score (0-10)"""
        # High mission clarity = more coherence
        # High employee satisfaction = more coherence
        # Low turnover = more coherence
        # Positive news sentiment = more coherence
        
        mission_factor = m.mission_clarity / 10 * 3  # 0-3
        satisfaction_factor = m.employee_satisfaction / 10 * 3  # 0-3
        turnover_factor = max(0, (1 - m.employee_turnover_pct / 50)) * 2  # 0-2
        sentiment_factor = (m.news_sentiment + 1) * 1  # 0-2 (from -1 to 1)
        
        return min(mission_factor + satisfaction_factor + turnover_factor + sentiment_factor, 10.0)
    
    def determine_stage(self, vectors: SPATVectors) -> Tuple[int, float]:
        """
        Determine SAP stage using nearest-neighbor in 5D space
        Returns: (stage_number, confidence_score)
        """
        min_distance = float('inf')
        best_stage = 0
        
        for stage, ideal_vectors in self.stage_weights.items():
            distance = self._calculate_vector_distance(vectors, ideal_vectors)
            if distance < min_distance:
                min_distance = distance
                best_stage = stage
        
        # Confidence = inverse of distance (normalized)
        confidence = max(0, (1 - min_distance / 10)) * 100
        
        return best_stage, confidence
    
    def _calculate_vector_distance(self, v1: SPATVectors, v2: SPATVectors) -> float:
        """Calculate Euclidean distance between two SPAT vectors"""
        return (
            (v1.complexity - v2.complexity) ** 2 +
            (v1.stability - v2.stability) ** 2 +
            (v1.tension - v2.tension) ** 2 +
            (v1.adaptability - v2.adaptability) ** 2 +
            (v1.coherence - v2.coherence) ** 2
        ) ** 0.5
    
    def detect_permanence_trap(self, metrics: CorporateMetrics, vectors: SPATVectors) -> Dict:
        """
        Detect Stage 8 permanence trap indicators
        Returns trap score and specific warnings
        """
        trap_indicators = []
        
        # Adaptability dropping
        if vectors.adaptability < 5.0:
            trap_indicators.append("Low adaptability (< 5.0)")
        
        # Bureaucratic bloat
        if metrics.management_layers > 8:
            trap_indicators.append(f"Bureaucratic ossification ({metrics.management_layers} layers)")
        
        # Innovation theater (high R&D, low output)
        if metrics.r_d_spending_pct > 15 and metrics.new_products_last_2yrs < 2:
            trap_indicators.append("Innovation theater (spending without results)")
        
        # Leadership stagnation
        if metrics.ceo_tenure_years > 12:
            trap_indicators.append(f"Leadership stagnation (CEO {metrics.ceo_tenure_years} years)")
        
        # Low growth despite high stability
        if vectors.stability > 8.0 and metrics.revenue_growth_yoy < 5:
            trap_indicators.append("Rigidity (high stability, low growth)")
        
        trap_score = len(trap_indicators)
        
        if trap_score >= 4:
            recommendation = "PERMANENCE TRAP CONFIRMED - Sell or short"
            risk_level = "CRITICAL"
        elif trap_score >= 2:
            recommendation = "PERMANENCE TRAP RISK - Avoid or exit"
            risk_level = "HIGH"
        else:
            recommendation = "No permanence trap detected"
            risk_level = "LOW"
        
        return {
            "trap_score": trap_score,
            "indicators": trap_indicators,
            "recommendation": recommendation,
            "risk_level": risk_level
        }
    
    def calculate_container_strength(self, metrics: CorporateMetrics, vectors: SPATVectors) -> Dict:
        """
        Container Rule: Can the structure hold the complexity?
        Critical for Stage 4 foundation companies
        """
        # Structure capacity = stability × coherence / bureaucracy
        structure_capacity = (
            vectors.stability * 
            vectors.coherence * 
            (1 / (1 + metrics.management_layers / 10))
        )
        
        # Complexity load = growth × complexity × tension
        complexity_load = (
            (1 + metrics.revenue_growth_yoy / 100) *
            vectors.complexity *
            (1 + vectors.tension / 10)
        )
        
        container_ratio = structure_capacity / complexity_load if complexity_load > 0 else 0
        
        if container_ratio < 0.5:
            status = "CONTAINER BREACH IMMINENT"
            recommendation = "Crisis within 12 months - Reduce complexity or strengthen foundation"
            risk_level = "CRITICAL"
        elif container_ratio < 0.8:
            status = "CONTAINER STRAIN"
            recommendation = "Strengthen foundation before scaling"
            risk_level = "ELEVATED"
        elif container_ratio > 1.2:
            status = "CONTAINER READY"
            recommendation = "Safe to add complexity/scale"
            risk_level = "LOW"
        else:
            status = "CONTAINER BALANCED"
            recommendation = "Optimal state - maintain equilibrium"
            risk_level = "LOW"
        
        return {
            "container_ratio": round(container_ratio, 2),
            "structure_capacity": round(structure_capacity, 2),
            "complexity_load": round(complexity_load, 2),
            "status": status,
            "recommendation": recommendation,
            "risk_level": risk_level
        }
    
    def generate_investment_signal(self, stage: int, metrics: CorporateMetrics, vectors: SPATVectors) -> Dict:
        """
        Generate investment recommendation based on SAP stage
        """
        signals = {
            0: {"action": "AVOID", "reason": "Too early - no product/revenue", "risk": "EXTREME"},
            1: {"action": "SPECULATIVE BUY", "reason": "High risk, high reward - assess founder adaptability", "risk": "VERY HIGH"},
            2: {"action": "BUY", "reason": "Series A sweet spot - product-market fit validated", "risk": "HIGH"},
            3: {"action": "MOMENTUM PLAY", "reason": "Explosive growth - ride momentum but watch for Stage 4 skip", "risk": "HIGH"},
            4: {"action": "STRONG BUY", "reason": "Best long-term bet - sustainable foundation", "risk": "LOW"},
            5: {"action": "HOLD/WATCH", "reason": "High volatility zone - outcome depends on adaptability", "risk": "VERY HIGH"},
            6: {"action": "HOLD", "reason": "Peak performance - hard to enter, hard to predict duration", "risk": "MEDIUM"},
            7: {"action": "AVOID/SHORT", "reason": "Crisis deepening - only buy if adaptability > 8.0", "risk": "EXTREME"},
            8: {"action": "SELL/SHORT", "reason": "Permanence trap - collapse likely within 5-10 years", "risk": "HIGH"},
            9: {"action": "N/A", "reason": "Typically not publicly traded - mission over profit", "risk": "N/A"},
        }
        
        base_signal = signals[stage]
        
        # Modify based on specific metrics
        if stage == 4:
            container = self.calculate_container_strength(metrics, vectors)
            if container["container_ratio"] < 0.8:
                base_signal["action"] = "HOLD"
                base_signal["reason"] += f" - Container strain detected ({container['container_ratio']})"
        
        if stage == 8:
            trap = self.detect_permanence_trap(metrics, vectors)
            base_signal["reason"] += f" - Trap score: {trap['trap_score']}/5"
        
        return base_signal
    
    def analyze_company(self, company_name: str, metrics: CorporateMetrics) -> Dict:
        """
        Complete SAP analysis of a company
        Returns comprehensive diagnostic report
        """
        # Calculate SPAT vectors
        vectors = self.calculate_spat_vectors(metrics)
        
        # Determine stage
        stage, confidence = self.determine_stage(vectors)
        
        # Generate investment signal
        signal = self.generate_investment_signal(stage, metrics, vectors)
        
        # Stage-specific analysis
        stage_specific = {}
        if stage == 8:
            stage_specific["permanence_trap"] = self.detect_permanence_trap(metrics, vectors)
        elif stage == 4:
            stage_specific["container_analysis"] = self.calculate_container_strength(metrics, vectors)
        
        return {
            "company": company_name,
            "sap_stage": stage,
            "stage_name": SAPStage(stage).name,
            "confidence": round(confidence, 1),
            "spat_vectors": {
                "complexity": round(vectors.complexity, 2),
                "stability": round(vectors.stability, 2),
                "tension": round(vectors.tension, 2),
                "adaptability": round(vectors.adaptability, 2),
                "coherence": round(vectors.coherence, 2)
            },
            "investment_signal": signal,
            "stage_specific_analysis": stage_specific
        }


# Example usage
if __name__ == "__main__":
    analyzer = CorporateSAPAnalyzer()
    
    # Example: Analyze a hypothetical Stage 8 trap company
    trap_company = CorporateMetrics(
        revenue_growth_yoy=3.0,
        profit_margin=25.0,
        debt_to_equity=0.5,
        cash_reserves_months=18.0,
        r_d_spending_pct=18.0,
        employee_turnover_pct=22.0,
        leadership_tenure_years=8.0,
        management_layers=12,
        employee_satisfaction=6.5,
        market_share_pct=35.0,
        market_growth_yoy=2.0,
        competitive_threats=8,
        new_products_last_2yrs=1,
        innovation_index=4.5,
        ceo_tenure_years=15.0,
        mission_clarity=7.0,
        news_sentiment=0.3,
        analyst_rating="hold"
    )
    
    result = analyzer.analyze_company("Legacy Corp", trap_company)
    print(json.dumps(result, indent=2))

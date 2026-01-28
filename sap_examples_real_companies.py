"""
SAP CORPORATE ANALYZER - REAL WORLD EXAMPLES
Demonstrates analysis of well-known companies across different SAP stages
"""

from sap_corporate_analyzer import CorporateSAPAnalyzer, CorporateMetrics
import json


def analyze_real_companies():
    """
    Analyze real companies to demonstrate SAP stage classification
    """
    analyzer = CorporateSAPAnalyzer()
    
    companies = {
        "NVIDIA (Stage 3 - Expression/Breakthrough)": CorporateMetrics(
            revenue_growth_yoy=126.0,  # AI boom
            profit_margin=55.0,
            debt_to_equity=0.3,
            cash_reserves_months=36.0,
            r_d_spending_pct=22.0,
            employee_turnover_pct=8.0,
            leadership_tenure_years=6.0,
            management_layers=5,
            employee_satisfaction=8.5,
            market_share_pct=85.0,  # AI chips
            market_growth_yoy=45.0,
            competitive_threats=4,
            new_products_last_2yrs=3,
            innovation_index=9.0,
            ceo_tenure_years=31.0,  # Jensen Huang
            mission_clarity=9.0,
            news_sentiment=0.8,
            analyst_rating="buy"
        ),
        
        "Microsoft (Stage 4 - Foundation)": CorporateMetrics(
            revenue_growth_yoy=16.0,
            profit_margin=42.0,
            debt_to_equity=0.4,
            cash_reserves_months=48.0,
            r_d_spending_pct=13.0,
            employee_turnover_pct=12.0,
            leadership_tenure_years=7.0,
            management_layers=7,
            employee_satisfaction=7.8,
            market_share_pct=28.0,  # Cloud
            market_growth_yoy=20.0,
            competitive_threats=3,
            new_products_last_2yrs=4,  # Copilot, Azure AI, etc.
            innovation_index=7.5,
            ceo_tenure_years=10.0,  # Satya Nadella
            mission_clarity=8.5,
            news_sentiment=0.6,
            analyst_rating="buy"
        ),
        
        "Intel (Stage 8 - Permanence Trap)": CorporateMetrics(
            revenue_growth_yoy=-14.0,  # Declining
            profit_margin=8.0,  # Compressed
            debt_to_equity=0.5,
            cash_reserves_months=24.0,
            r_d_spending_pct=25.0,  # High spending, low output
            employee_turnover_pct=18.0,
            leadership_tenure_years=9.0,
            management_layers=10,
            employee_satisfaction=6.2,
            market_share_pct=15.0,  # Lost to AMD, NVIDIA
            market_growth_yoy=8.0,
            competitive_threats=6,
            new_products_last_2yrs=1,  # Few meaningful innovations
            innovation_index=4.0,
            ceo_tenure_years=3.0,  # Pat Gelsinger (turnaround attempt)
            mission_clarity=6.5,
            news_sentiment=-0.2,
            analyst_rating="hold"
        ),
        
        "Tesla (Stage 5 - Threshold)": CorporateMetrics(
            revenue_growth_yoy=19.0,  # Slowing from peak
            profit_margin=9.2,
            debt_to_equity=0.1,
            cash_reserves_months=18.0,
            r_d_spending_pct=3.5,  # Low for tech
            employee_turnover_pct=28.0,  # High
            leadership_tenure_years=5.0,
            management_layers=6,
            employee_satisfaction=6.8,
            market_share_pct=55.0,  # EV market
            market_growth_yoy=25.0,
            competitive_threats=12,  # Everyone entering EV
            new_products_last_2yrs=2,  # Cybertruck, FSD
            innovation_index=7.0,
            ceo_tenure_years=21.0,  # Elon Musk
            mission_clarity=8.0,
            news_sentiment=0.1,  # Mixed
            analyst_rating="hold"
        ),
        
        "Boeing (Stage 7 - Crisis)": CorporateMetrics(
            revenue_growth_yoy=-8.0,
            profit_margin=-5.0,  # Losses
            debt_to_equity=2.8,  # High debt
            cash_reserves_months=12.0,
            r_d_spending_pct=3.0,  # Cut during crisis
            employee_turnover_pct=35.0,  # Mass exodus
            leadership_tenure_years=4.0,
            management_layers=9,
            employee_satisfaction=4.5,  # Cultural collapse
            market_share_pct=40.0,  # Duopoly with Airbus
            market_growth_yoy=5.0,
            competitive_threats=2,
            new_products_last_2yrs=0,  # 737 MAX crisis
            innovation_index=3.0,
            ceo_tenure_years=5.0,  # Dave Calhoun
            mission_clarity=5.0,
            news_sentiment=-0.6,  # Very negative
            analyst_rating="sell"
        ),
        
        "Zoom (Stage 4 - Foundation, post-pandemic)": CorporateMetrics(
            revenue_growth_yoy=3.0,  # Normalized after COVID boom
            profit_margin=28.0,
            debt_to_equity=0.0,  # No debt
            cash_reserves_months=60.0,  # Massive cash
            r_d_spending_pct=12.0,
            employee_turnover_pct=15.0,
            leadership_tenure_years=6.0,
            management_layers=5,
            employee_satisfaction=7.5,
            market_share_pct=45.0,  # Video conferencing
            market_growth_yoy=8.0,
            competitive_threats=5,  # Teams, Meet, etc.
            new_products_last_2yrs=3,  # Zoom Phone, Events, etc.
            innovation_index=6.5,
            ceo_tenure_years=14.0,  # Eric Yuan
            mission_clarity=8.0,
            news_sentiment=0.4,
            analyst_rating="hold"
        ),
    }
    
    print("=" * 80)
    print("SAP CORPORATE ANALYZER - REAL WORLD EXAMPLES")
    print("=" * 80)
    print()
    
    for company_name, metrics in companies.items():
        result = analyzer.analyze_company(company_name, metrics)
        
        print(f"\n{'=' * 80}")
        print(f"COMPANY: {result['company']}")
        print(f"{'=' * 80}")
        print(f"SAP STAGE: {result['sap_stage']} - {result['stage_name']}")
        print(f"CONFIDENCE: {result['confidence']}%")
        print()
        print("SPAT VECTORS:")
        for vector, value in result['spat_vectors'].items():
            print(f"  {vector.capitalize():15} {value:5.2f}")
        print()
        print("INVESTMENT SIGNAL:")
        print(f"  Action: {result['investment_signal']['action']}")
        print(f"  Reason: {result['investment_signal']['reason']}")
        print(f"  Risk:   {result['investment_signal']['risk']}")
        
        if result['stage_specific_analysis']:
            print()
            print("STAGE-SPECIFIC ANALYSIS:")
            for key, value in result['stage_specific_analysis'].items():
                if key == "permanence_trap":
                    print(f"\n  PERMANENCE TRAP DETECTION:")
                    print(f"    Score: {value['trap_score']}/5")
                    print(f"    Risk:  {value['risk_level']}")
                    print(f"    Recommendation: {value['recommendation']}")
                    if value['indicators']:
                        print(f"    Indicators:")
                        for indicator in value['indicators']:
                            print(f"      - {indicator}")
                elif key == "container_analysis":
                    print(f"\n  CONTAINER RULE ANALYSIS:")
                    print(f"    Ratio:  {value['container_ratio']}")
                    print(f"    Status: {value['status']}")
                    print(f"    Risk:   {value['risk_level']}")
                    print(f"    Recommendation: {value['recommendation']}")
        
        print()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_real_companies()

"""
LUMINARK AUTOMATED ANALYZER
Connects data connectors to SAP analyzer for fully automated corporate analysis
"""

from luminark_data_connectors import CorporateDataAggregator
from sap_corporate_analyzer import CorporateSAPAnalyzer, CorporateMetrics
import json
from typing import Dict, List, Optional
from datetime import datetime


class AutomatedSAPAnalyzer:
    """
    Fully automated SAP analysis using real-world data
    """
    
    def __init__(self,
                 alpha_vantage_key: Optional[str] = None,
                 news_api_key: Optional[str] = None,
                 sec_email: Optional[str] = None):
        
        self.data_aggregator = CorporateDataAggregator(
            alpha_vantage_key=alpha_vantage_key,
            news_api_key=news_api_key,
            sec_email=sec_email
        )
        self.sap_analyzer = CorporateSAPAnalyzer()
    
    def analyze_company_auto(self, symbol: str, company_name: Optional[str] = None) -> Dict:
        """
        Fully automated analysis: Gather data → Calculate metrics → SAP analysis
        """
        print(f"\n{'='*80}")
        print(f"LUMINARK AUTOMATED SAP ANALYSIS: {symbol}")
        print(f"{'='*80}\n")
        
        # Step 1: Gather real-world data
        raw_data = self.data_aggregator.gather_company_data(symbol, company_name)
        
        # Step 2: Convert to CorporateMetrics
        metrics = self._convert_to_metrics(raw_data)
        
        if not metrics:
            return {
                'error': 'Insufficient data to perform analysis',
                'symbol': symbol
            }
        
        # Step 3: Perform SAP analysis
        print("\n🧠 Performing SAP analysis...")
        sap_result = self.sap_analyzer.analyze_company(company_name or symbol, metrics)
        
        # Step 4: Combine results
        complete_analysis = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timestamp': datetime.now().isoformat(),
            'raw_data_sources': list(raw_data['sources'].keys()),
            'sap_analysis': sap_result,
            'data_quality_score': self._calculate_data_quality(raw_data)
        }
        
        return complete_analysis
    
    def _convert_to_metrics(self, raw_data: Dict) -> Optional[CorporateMetrics]:
        """
        Convert raw data from multiple sources into CorporateMetrics
        Uses intelligent defaults and estimations when data is missing
        """
        normalized = raw_data.get('normalized_metrics', {})
        yahoo = raw_data.get('sources', {}).get('yahoo_finance', {})
        alpha = raw_data.get('sources', {}).get('alpha_vantage', {})
        
        # Extract what we have
        revenue_growth = normalized.get('revenue_growth_yoy', 0)
        profit_margin = normalized.get('profit_margin', 0)
        news_sentiment = normalized.get('news_sentiment', 0)
        
        # Estimate missing metrics using industry averages and correlations
        # (In production, you'd have a database of industry benchmarks)
        
        # Debt to equity: estimate from profit margin
        # High margin companies tend to have lower debt
        debt_to_equity = max(0, 1.0 - (profit_margin / 50)) if profit_margin > 0 else 0.8
        
        # Cash reserves: estimate from market cap and profit margin
        market_cap = alpha.get('market_cap', 0)
        if market_cap > 1e12:  # >$1T
            cash_reserves_months = 48
        elif market_cap > 1e11:  # >$100B
            cash_reserves_months = 36
        elif market_cap > 1e10:  # >$10B
            cash_reserves_months = 24
        else:
            cash_reserves_months = 12
        
        # R&D spending: estimate from industry
        sector = yahoo.get('sector', 'Unknown')
        if sector in ['Technology', 'Healthcare']:
            r_d_spending_pct = 15.0
        elif sector in ['Consumer Cyclical', 'Industrials']:
            r_d_spending_pct = 5.0
        else:
            r_d_spending_pct = 8.0
        
        # Employee metrics: estimates based on company size
        employee_count = yahoo.get('employee_count', 0)
        if employee_count > 100000:
            employee_turnover_pct = 15.0
            management_layers = 9
            employee_satisfaction = 7.0
        elif employee_count > 10000:
            employee_turnover_pct = 12.0
            management_layers = 7
            employee_satisfaction = 7.5
        else:
            employee_turnover_pct = 18.0
            management_layers = 5
            employee_satisfaction = 7.2
        
        # Market position: estimate from market cap
        if market_cap > 5e11:  # >$500B
            market_share_pct = 30.0
            competitive_threats = 3
        elif market_cap > 1e11:  # >$100B
            market_share_pct = 20.0
            competitive_threats = 5
        else:
            market_share_pct = 10.0
            competitive_threats = 8
        
        # Innovation metrics: correlate with R&D and revenue growth
        if revenue_growth > 30:
            new_products_last_2yrs = 4
            innovation_index = 8.5
        elif revenue_growth > 15:
            new_products_last_2yrs = 2
            innovation_index = 7.0
        else:
            new_products_last_2yrs = 1
            innovation_index = 5.5
        
        # Leadership: estimates
        leadership_tenure_years = 6.0
        ceo_tenure_years = 8.0
        mission_clarity = 7.5
        
        # Market growth: estimate from sector
        if sector == 'Technology':
            market_growth_yoy = 15.0
        elif sector == 'Healthcare':
            market_growth_yoy = 8.0
        else:
            market_growth_yoy = 5.0
        
        # Analyst rating: derive from sentiment and growth
        if news_sentiment > 0.3 and revenue_growth > 20:
            analyst_rating = "buy"
        elif news_sentiment < -0.3 or revenue_growth < 0:
            analyst_rating = "sell"
        else:
            analyst_rating = "hold"
        
        try:
            return CorporateMetrics(
                revenue_growth_yoy=revenue_growth,
                profit_margin=profit_margin,
                debt_to_equity=debt_to_equity,
                cash_reserves_months=cash_reserves_months,
                r_d_spending_pct=r_d_spending_pct,
                employee_turnover_pct=employee_turnover_pct,
                leadership_tenure_years=leadership_tenure_years,
                management_layers=management_layers,
                employee_satisfaction=employee_satisfaction,
                market_share_pct=market_share_pct,
                market_growth_yoy=market_growth_yoy,
                competitive_threats=competitive_threats,
                new_products_last_2yrs=new_products_last_2yrs,
                innovation_index=innovation_index,
                ceo_tenure_years=ceo_tenure_years,
                mission_clarity=mission_clarity,
                news_sentiment=news_sentiment,
                analyst_rating=analyst_rating
            )
        except Exception as e:
            print(f"Error creating CorporateMetrics: {e}")
            return None
    
    def _calculate_data_quality(self, raw_data: Dict) -> float:
        """
        Calculate data quality score (0-100)
        Higher score = more data sources available
        """
        sources = raw_data.get('sources', {})
        score = 0
        
        if 'yahoo_finance' in sources and sources['yahoo_finance']:
            score += 40
        if 'alpha_vantage' in sources and sources['alpha_vantage']:
            score += 30
        if 'news_sentiment' in sources:
            score += 20
        if 'sec_filings' in sources:
            score += 10
        
        return min(score, 100)
    
    def batch_analyze(self, symbols: List[str], company_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Analyze multiple companies in batch
        """
        results = []
        
        if not company_names:
            company_names = [None] * len(symbols)
        
        for symbol, name in zip(symbols, company_names):
            try:
                result = self.analyze_company_auto(symbol, name)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return results
    
    def generate_report(self, analysis: Dict) -> str:
        """
        Generate human-readable report from analysis
        """
        if 'error' in analysis:
            return f"ERROR: {analysis['error']}"
        
        sap = analysis['sap_analysis']
        
        report = f"""
{'='*80}
LUMINARK SAP CORPORATE ANALYSIS REPORT
{'='*80}

Company: {analysis['company_name']} ({analysis['symbol']})
Analysis Date: {analysis['timestamp'][:10]}
Data Quality: {analysis['data_quality_score']}/100

{'='*80}
SAP CLASSIFICATION
{'='*80}

Stage: {sap['sap_stage']} - {sap['stage_name']}
Confidence: {sap['confidence']}%

SPAT Vectors:
  Complexity:    {sap['spat_vectors']['complexity']:.2f}/10
  Stability:     {sap['spat_vectors']['stability']:.2f}/10
  Tension:       {sap['spat_vectors']['tension']:.2f}/10
  Adaptability:  {sap['spat_vectors']['adaptability']:.2f}/10
  Coherence:     {sap['spat_vectors']['coherence']:.2f}/10

{'='*80}
INVESTMENT SIGNAL
{'='*80}

Action: {sap['investment_signal']['action']}
Risk Level: {sap['investment_signal']['risk']}

Rationale:
{sap['investment_signal']['reason']}

"""
        
        # Add stage-specific analysis
        if sap.get('stage_specific_analysis'):
            report += f"{'='*80}\n"
            report += "STAGE-SPECIFIC ANALYSIS\n"
            report += f"{'='*80}\n\n"
            
            for key, value in sap['stage_specific_analysis'].items():
                if key == 'permanence_trap':
                    report += "PERMANENCE TRAP DETECTION:\n"
                    report += f"  Score: {value['trap_score']}/5\n"
                    report += f"  Risk: {value['risk_level']}\n"
                    report += f"  Recommendation: {value['recommendation']}\n"
                    if value['indicators']:
                        report += "  Indicators:\n"
                        for indicator in value['indicators']:
                            report += f"    - {indicator}\n"
                
                elif key == 'container_analysis':
                    report += "CONTAINER RULE ANALYSIS:\n"
                    report += f"  Ratio: {value['container_ratio']}\n"
                    report += f"  Status: {value['status']}\n"
                    report += f"  Risk: {value['risk_level']}\n"
                    report += f"  Recommendation: {value['recommendation']}\n"
        
        report += f"\n{'='*80}\n"
        report += "END OF REPORT\n"
        report += f"{'='*80}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize automated analyzer
    # For demo, we'll use Yahoo Finance only (no API keys needed)
    analyzer = AutomatedSAPAnalyzer()
    
    # Analyze a single company
    print("\n🚀 LUMINARK AUTOMATED SAP ANALYZER - DEMO\n")
    
    # Note: This will work with Yahoo Finance data only
    # For full functionality, add API keys
    result = analyzer.analyze_company_auto("MSFT", "Microsoft")
    
    # Generate and print report
    report = analyzer.generate_report(result)
    print(report)
    
    # Save results
    with open(f"luminark_analysis_{result['symbol']}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Full analysis saved to luminark_analysis_{result['symbol']}.json")

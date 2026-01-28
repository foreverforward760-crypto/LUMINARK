"""
LUMINARK DATA CONNECTORS
Real-world data integration for SAP Corporate Analyzer

Connects to free/low-cost APIs to automatically gather corporate metrics
"""

import requests
import json
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
from dataclasses import asdict


class DataConnector:
    """Base class for data connectors"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}
        self.rate_limit_delay = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Simple rate limiting"""
        time.sleep(self.rate_limit_delay)
    
    def _cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{data_type}_{datetime.now().strftime('%Y%m%d')}"


class YahooFinanceConnector(DataConnector):
    """
    Yahoo Finance API - FREE
    Provides: Stock prices, financial statements, key metrics
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://query2.finance.yahoo.com/v10/finance"
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get basic company information"""
        cache_key = self._cache_key(symbol, "info")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"{self.base_url}/quoteSummary/{symbol}"
            params = {
                "modules": "assetProfile,financialData,defaultKeyStatistics,incomeStatementHistory"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = data
            self._rate_limit()
            return data
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return {}
    
    def extract_metrics(self, symbol: str) -> Dict:
        """Extract relevant metrics from Yahoo Finance data"""
        data = self.get_company_info(symbol)
        
        if not data or 'quoteSummary' not in data:
            return {}
        
        result = data['quoteSummary']['result'][0] if data['quoteSummary']['result'] else {}
        
        financial = result.get('financialData', {})
        key_stats = result.get('defaultKeyStatistics', {})
        profile = result.get('assetProfile', {})
        
        return {
            'revenue_growth_yoy': financial.get('revenueGrowth', {}).get('raw', 0) * 100,
            'profit_margin': financial.get('profitMargins', {}).get('raw', 0) * 100,
            'debt_to_equity': financial.get('debtToEquity', {}).get('raw', 0) / 100,
            'current_ratio': financial.get('currentRatio', {}).get('raw', 1.0),
            'employee_count': profile.get('fullTimeEmployees', 0),
            'sector': profile.get('sector', 'Unknown'),
            'industry': profile.get('industry', 'Unknown'),
        }


class SECEdgarConnector(DataConnector):
    """
    SEC EDGAR API - FREE
    Provides: Official financial filings (10-K, 10-Q, 8-K)
    """
    
    def __init__(self, email: str):
        super().__init__()
        self.base_url = "https://data.sec.gov"
        self.headers = {
            'User-Agent': f'LUMINARK SAP Analyzer contact@luminark.ai ({email})'
        }
        self.rate_limit_delay = 0.1  # SEC requires 10 requests/second max
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a ticker"""
        try:
            url = f"{self.base_url}/submissions/CIK{ticker.upper()}.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json().get('cik')
            return None
            
        except Exception as e:
            print(f"Error fetching CIK for {ticker}: {e}")
            return None
    
    def get_recent_filings(self, cik: str, form_type: str = "10-K") -> List[Dict]:
        """Get recent filings for a company"""
        try:
            url = f"{self.base_url}/submissions/CIK{cik.zfill(10)}.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings = data.get('filings', {}).get('recent', {})
            
            # Filter by form type
            results = []
            for i, form in enumerate(filings.get('form', [])):
                if form == form_type:
                    results.append({
                        'form': form,
                        'filing_date': filings['filingDate'][i],
                        'accession_number': filings['accessionNumber'][i]
                    })
            
            self._rate_limit()
            return results[:5]  # Return 5 most recent
            
        except Exception as e:
            print(f"Error fetching filings for CIK {cik}: {e}")
            return []


class AlphaVantageConnector(DataConnector):
    """
    Alpha Vantage API - FREE (500 requests/day)
    Provides: Financial statements, earnings, company overview
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12.0  # Free tier: 5 requests/minute
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company overview with key metrics"""
        cache_key = self._cache_key(symbol, "overview")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = data
            self._rate_limit()
            return data
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return {}
    
    def extract_metrics(self, symbol: str) -> Dict:
        """Extract metrics from Alpha Vantage overview"""
        data = self.get_company_overview(symbol)
        
        if not data or 'Symbol' not in data:
            return {}
        
        def safe_float(value, default=0.0):
            try:
                return float(value) if value and value != 'None' else default
            except (ValueError, TypeError):
                return default
        
        return {
            'market_cap': safe_float(data.get('MarketCapitalization', 0)),
            'pe_ratio': safe_float(data.get('PERatio', 0)),
            'peg_ratio': safe_float(data.get('PEGRatio', 0)),
            'book_value': safe_float(data.get('BookValue', 0)),
            'dividend_yield': safe_float(data.get('DividendYield', 0)) * 100,
            'profit_margin': safe_float(data.get('ProfitMargin', 0)) * 100,
            'operating_margin': safe_float(data.get('OperatingMarginTTM', 0)) * 100,
            'roe': safe_float(data.get('ReturnOnEquityTTM', 0)) * 100,
            'revenue_ttm': safe_float(data.get('RevenueTTM', 0)),
            'quarterly_revenue_growth': safe_float(data.get('QuarterlyRevenueGrowthYOY', 0)) * 100,
            'analyst_target_price': safe_float(data.get('AnalystTargetPrice', 0)),
        }


class NewsAPIConnector(DataConnector):
    """
    News API - FREE (100 requests/day)
    Provides: News sentiment analysis
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2"
    
    def get_company_news(self, company_name: str, days_back: int = 7) -> List[Dict]:
        """Get recent news articles about a company"""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': company_name,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'language': 'en'
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self._rate_limit()
            return data.get('articles', [])
            
        except Exception as e:
            print(f"Error fetching news for {company_name}: {e}")
            return []
    
    def calculate_sentiment(self, company_name: str) -> float:
        """
        Calculate news sentiment score (-1 to 1)
        Simple keyword-based sentiment (can be enhanced with NLP)
        """
        articles = self.get_company_news(company_name)
        
        if not articles:
            return 0.0
        
        positive_words = ['growth', 'profit', 'success', 'innovation', 'breakthrough', 
                         'record', 'strong', 'beat', 'exceeds', 'surge']
        negative_words = ['loss', 'decline', 'crisis', 'lawsuit', 'investigation',
                         'layoff', 'miss', 'weak', 'concern', 'struggle']
        
        sentiment_scores = []
        for article in articles[:20]:  # Analyze top 20 articles
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
                sentiment_scores.append(score)
        
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0


class CorporateDataAggregator:
    """
    Aggregates data from multiple sources to build complete corporate profile
    """
    
    def __init__(self, 
                 alpha_vantage_key: Optional[str] = None,
                 news_api_key: Optional[str] = None,
                 sec_email: Optional[str] = None):
        
        self.yahoo = YahooFinanceConnector()
        self.alpha_vantage = AlphaVantageConnector(alpha_vantage_key) if alpha_vantage_key else None
        self.news_api = NewsAPIConnector(news_api_key) if news_api_key else None
        self.sec = SECEdgarConnector(sec_email) if sec_email else None
    
    def gather_company_data(self, symbol: str, company_name: Optional[str] = None) -> Dict:
        """
        Gather comprehensive company data from all available sources
        """
        print(f"\n🔍 Gathering data for {symbol}...")
        
        data = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Yahoo Finance (always available)
        print("  📊 Fetching Yahoo Finance data...")
        yahoo_data = self.yahoo.extract_metrics(symbol)
        data['sources']['yahoo_finance'] = yahoo_data
        
        # Alpha Vantage (if API key provided)
        if self.alpha_vantage:
            print("  📈 Fetching Alpha Vantage data...")
            av_data = self.alpha_vantage.extract_metrics(symbol)
            data['sources']['alpha_vantage'] = av_data
        
        # News sentiment (if API key provided)
        if self.news_api and company_name:
            print("  📰 Analyzing news sentiment...")
            sentiment = self.news_api.calculate_sentiment(company_name)
            data['sources']['news_sentiment'] = {
                'sentiment_score': sentiment,
                'interpretation': self._interpret_sentiment(sentiment)
            }
        
        # SEC filings (if email provided)
        if self.sec:
            print("  📄 Fetching SEC filings...")
            cik = self.sec.get_company_cik(symbol)
            if cik:
                filings = self.sec.get_recent_filings(cik)
                data['sources']['sec_filings'] = {
                    'cik': cik,
                    'recent_10k': filings
                }
        
        # Merge and normalize data
        data['normalized_metrics'] = self._normalize_metrics(data['sources'])
        
        print(f"  ✅ Data gathering complete for {symbol}")
        return data
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
        if score > 0.3:
            return "Positive"
        elif score < -0.3:
            return "Negative"
        else:
            return "Neutral"
    
    def _normalize_metrics(self, sources: Dict) -> Dict:
        """
        Normalize and merge metrics from different sources
        Prioritize more reliable sources
        """
        normalized = {}
        
        # Priority: Alpha Vantage > Yahoo Finance
        yahoo = sources.get('yahoo_finance', {})
        alpha = sources.get('alpha_vantage', {})
        news = sources.get('news_sentiment', {})
        
        # Revenue growth
        normalized['revenue_growth_yoy'] = (
            alpha.get('quarterly_revenue_growth') or 
            yahoo.get('revenue_growth_yoy', 0)
        )
        
        # Profit margin
        normalized['profit_margin'] = (
            alpha.get('profit_margin') or 
            yahoo.get('profit_margin', 0)
        )
        
        # Debt to equity
        normalized['debt_to_equity'] = yahoo.get('debt_to_equity', 0)
        
        # News sentiment
        normalized['news_sentiment'] = news.get('sentiment_score', 0.0)
        
        # Market data
        normalized['market_cap'] = alpha.get('market_cap', 0)
        normalized['pe_ratio'] = alpha.get('pe_ratio', 0)
        
        return normalized
    
    def save_to_file(self, data: Dict, filename: Optional[str] = None):
        """Save gathered data to JSON file"""
        if not filename:
            filename = f"data/{data['symbol']}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize aggregator (add your API keys here)
    aggregator = CorporateDataAggregator(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY",  # Get free at alphavantage.co
        news_api_key="YOUR_NEWS_API_KEY",            # Get free at newsapi.org
        sec_email="your@email.com"                    # Required for SEC API
    )
    
    # Gather data for a company
    data = aggregator.gather_company_data("MSFT", "Microsoft")
    
    # Print results
    print("\n" + "="*80)
    print("NORMALIZED METRICS:")
    print("="*80)
    for key, value in data['normalized_metrics'].items():
        print(f"{key:25} {value}")
    
    # Save to file
    # aggregator.save_to_file(data)

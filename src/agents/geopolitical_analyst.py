"""
Geopolitical Analyst Agent

This agent analyzes geopolitical news and events to score ETFs based on
geopolitical risks and opportunities.
"""

from src.agents.base_agent import BaseAgent
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class GeopoliticalAnalystAgent(BaseAgent):
    """
    Geopolitical Analyst Agent that analyzes news and geopolitical events.
    
    This agent focuses on:
    - Geopolitical news and events
    - Regional risks and opportunities
    - Currency and trade impacts
    - Country-specific ETF analysis
    - Scoring ETFs from -1 (sell) to 1 (buy) based on geopolitical factors
    """
    
    def __init__(self, agent_name: str = "GeopoliticalAnalystAgent"):
        """Initialize the geopolitical analyst agent."""
        super().__init__(agent_name)
        self.specialization = "geopolitical_analysis"
        self.analysis_focus = "news_and_events"
    
    def analyze(self, state: dict) -> dict:
        """
        Analyze geopolitical news and score ETFs.
        
        Args:
            state: LangGraph state dictionary containing:
                - news: Geopolitical news data
                - universe: List of ETFs to analyze
                - analyst_scores: Dictionary to store scores
                
        Returns:
            Updated state dictionary with geopolitical analyst scores
        """
        try:
            # Extract data from state
            news_data = state.get('news', [])
            universe = state.get('universe', [])
            
            # Ensure analyst_scores exists in state
            if 'analyst_scores' not in state:
                state['analyst_scores'] = {}
            
            # Create analysis prompt
            prompt = self._create_geopolitical_analysis_prompt(news_data, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse scores from response
            scores = self._parse_scores(response, universe)
            
            # Store scores in state
            state['analyst_scores']['geo'] = scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['geopolitical_analyst'] = {
                'scores': scores,
                'reasoning': f"Geopolitical analysis based on {len(news_data)} news articles and {len(universe)} ETFs",
                'key_factors': [article.get('title', 'Unknown')[:50] for article in news_data[:3]] if news_data else ['No news data available'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Geopolitical analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Geopolitical analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: 0.0 for etf in state.get('universe', [])}
            state['analyst_scores']['geo'] = neutral_scores
            return state
    
    def _create_geopolitical_analysis_prompt(self, news_data: list, universe: list) -> str:
        """
        Create a prompt for geopolitical analysis with comprehensive news coverage.
        
        Args:
            news_data: List of news articles (can be 50-100+ articles)
            universe: List of ETFs to analyze
            
        Returns:
            Formatted prompt string
        """
        # Categorize and format news data
        news_summary = self._format_news_data_comprehensive(news_data)
        
        prompt = f"""
        As a geopolitical analyst, analyze the comprehensive news from the past 30 days and score each ETF from -1 (strong sell) to 1 (strong buy).
        
        You have access to {len(news_data)} news articles covering major global events. Synthesize the key themes and their market implications.
        
        COMPREHENSIVE GEOPOLITICAL NEWS (Last 30 Days):
        {news_summary}
        
        ANALYSIS FRAMEWORK - Identify Major Themes:
        1. Regional Risks: Which countries/regions face heightened geopolitical risks?
           - Conflicts, political instability, regime changes
           - Regulatory crackdowns and policy shifts
           
        2. Trade & Supply Chains: 
           - Trade tensions, tariffs, sanctions
           - Supply chain disruptions
           - Reshoring vs offshoring trends
           
        3. Monetary Policy Divergence:
           - Central bank policy differences across regions
           - Currency volatility and competitive devaluations
           - Impact on currency ETFs (UUP, FXE, FXY, FXB, FXC, FXA, etc.)
           
        4. Commodity & Energy:
           - Oil/gas supply disruptions
           - Strategic resource competition
           - Impact on commodity ETFs (GLD, SLV, USO, UNG, DBC, etc.)
           
        5. Regional Economic Outlook:
           - China: Property sector, stimulus, US relations
           - Europe: Energy crisis, recession risk, ECB policy
           - Emerging Markets: Currency crises, debt stress
           - US: Political gridlock, fiscal policy
           - Impact on country-specific ETFs (EWJ, EWG, FXI, EWZ, INDA, etc.)
           
        6. Safe Haven Flows:
           - Flight to safety vs risk-on sentiment
           - Gold, treasuries, dollar strength
           - Impact on TLT, IEF, BND, GLD
        
        HISTORICAL CONTEXT FOR ANALYSIS:
        Consider how current events relate to major geopolitical patterns from the past 25 years:
        - 2000-2002: Dot-com crash, 9/11 attacks, Afghanistan War
        - 2003-2007: Iraq War, emerging market growth, commodity supercycle
        - 2008-2009: Global Financial Crisis, quantitative easing era begins
        - 2010-2015: European debt crisis, Arab Spring, China's rise
        - 2016-2020: Brexit, US-China trade war, COVID-19 pandemic
        - 2021-2025: Post-COVID recovery, Russia-Ukraine war, inflation surge, deglobalization
        
        SCORING INSTRUCTIONS:
        - Synthesize ALL {len(news_data)} articles to identify the dominant geopolitical themes
        - Compare current events to historical precedents and cycles
        - Score each ETF from -1.0 (strong sell) to 1.0 (strong buy)
        - Base scores on:
          * Severity and persistence of geopolitical risks (vs historical norms)
          * Regional exposure to conflicts/crises (learning from past crises)
          * Safe haven characteristics (how they performed in past crises)
          * Currency and commodity exposure (historical volatility patterns)
        - Consider both immediate shocks and longer-term structural shifts
        - Use 25-year historical perspective to assess if current risks are cyclical or structural
        - Differentiate scores meaningfully - not all ETFs should be neutral
        
        ETFs TO SCORE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores (differentiated, not all zeros):
        {{"SPY": 0.1, "EWJ": -0.2, "GLD": 0.3, "TLT": 0.2, "FXI": -0.4, ...}}
        """
        return prompt
    
    def _format_news_data(self, news_data: list) -> str:
        """Format news data for the prompt (legacy method)."""
        if not news_data:
            return "No geopolitical news available"
        
        formatted = []
        for i, article in enumerate(news_data[:10]):  # Limit to first 10 articles
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary')
            sentiment = article.get('sentiment', 'neutral')
            
            formatted.append(f"{i+1}. {title}")
            formatted.append(f"   Summary: {summary[:200]}...")
            formatted.append(f"   Sentiment: {sentiment}")
            formatted.append("")
        
        return '\n'.join(formatted) if formatted else "No geopolitical news available"
    
    def _format_news_data_comprehensive(self, news_data: list) -> str:
        """
        Format comprehensive news data by categorizing articles into themes.
        This helps the LLM process large volumes of news more effectively.
        """
        if not news_data:
            return "No geopolitical news available"
        
        # Categorize articles by keywords
        categories = {
            'china': [],
            'europe': [],
            'us_domestic': [],
            'trade_tariffs': [],
            'central_banks': [],
            'conflicts_geopolitics': [],
            'commodities_energy': [],
            'emerging_markets': [],
            'currencies': [],
            'other': []
        }
        
        for article in news_data:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            content = title + ' ' + summary
            
            # Categorize based on keywords
            if any(word in content for word in ['china', 'chinese', 'beijing', 'xi jinping']):
                categories['china'].append(article)
            elif any(word in content for word in ['europe', 'eu', 'eurozone', 'germany', 'france', 'ecb']):
                categories['europe'].append(article)
            elif any(word in content for word in ['tariff', 'trade war', 'trade tension', 'sanctions', 'export']):
                categories['trade_tariffs'].append(article)
            elif any(word in content for word in ['fed', 'federal reserve', 'interest rate', 'monetary policy', 'central bank']):
                categories['central_banks'].append(article)
            elif any(word in content for word in ['war', 'conflict', 'military', 'tension', 'geopolitical']):
                categories['conflicts_geopolitics'].append(article)
            elif any(word in content for word in ['oil', 'crude', 'energy', 'gas', 'opec', 'commodity']):
                categories['commodities_energy'].append(article)
            elif any(word in content for word in ['emerging market', 'brazil', 'india', 'mexico', 'africa']):
                categories['emerging_markets'].append(article)
            elif any(word in content for word in ['dollar', 'currency', 'forex', 'exchange rate', 'yen', 'euro']):
                categories['currencies'].append(article)
            elif any(word in content for word in ['us ', 'united states', 'america', 'congress', 'biden', 'trump']):
                categories['us_domestic'].append(article)
            else:
                categories['other'].append(article)
        
        # Format categorized news
        formatted = []
        formatted.append(f"=== COMPREHENSIVE NEWS ANALYSIS ({len(news_data)} Articles from Past 30 Days) ===\n")
        
        for category, articles in categories.items():
            if articles:
                category_name = category.replace('_', ' ').title()
                formatted.append(f"\n{category_name} ({len(articles)} articles):")
                
                # Show top 3-5 articles per category
                for i, article in enumerate(articles[:5], 1):
                    title = article.get('title', 'No title')
                    sentiment = article.get('sentiment', 'neutral')
                    formatted.append(f"  {i}. {title} [Sentiment: {sentiment}]")
                
                if len(articles) > 5:
                    formatted.append(f"  ... and {len(articles) - 5} more {category_name} articles")
        
        # Add summary statistics
        formatted.append(f"\n=== KEY THEMES SUMMARY ===")
        formatted.append(f"Total articles analyzed: {len(news_data)}")
        for category, articles in categories.items():
            if articles:
                category_name = category.replace('_', ' ').title()
                formatted.append(f"  - {category_name}: {len(articles)} articles")
        
        return '\n'.join(formatted)
    
    def _parse_scores(self, response: str, universe: list) -> dict:
        """
        Parse ETF scores from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to score
            
        Returns:
            Dictionary with ETF scores
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                scores = json.loads(json_str)
                
                # Validate scores
                validated_scores = {}
                for etf in universe:
                    if etf in scores:
                        score = float(scores[etf])
                        # Clamp score between -1 and 1
                        validated_scores[etf] = max(-1.0, min(1.0, score))
                    else:
                        validated_scores[etf] = 0.0
                
                return validated_scores
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: 0.0 for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse scores from response: {e}")
            return {etf: 0.0 for etf in universe}


# Example usage and testing
if __name__ == "__main__":
    print("Geopolitical Analyst Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = GeopoliticalAnalystAgent("TestGeopoliticalAnalyst")
        print(f"✓ Geopolitical analyst agent initialized")
        print(f"  Specialization: {agent.specialization}")
        print(f"  Provider: {agent.get_provider_info()['provider']}")
        
        # Test with sample state
        sample_state = {
            'news': [
                {
                    'title': 'Trade Tensions Rise Between US and China',
                    'summary': 'New tariffs announced on Chinese goods affecting global supply chains',
                    'sentiment': 'negative'
                },
                {
                    'title': 'European Central Bank Maintains Dovish Stance',
                    'summary': 'ECB keeps interest rates low to support economic recovery',
                    'sentiment': 'positive'
                }
            ],
            'universe': ['SPY', 'FXI', 'EWG', 'GLD'],
            'analyst_scores': {}
        }
        
        # Test analysis
        result_state = agent.analyze(sample_state)
        print(f"✓ Analysis completed")
        print(f"  Scores: {result_state.get('analyst_scores', {}).get('geo', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Geopolitical analyst test completed!")

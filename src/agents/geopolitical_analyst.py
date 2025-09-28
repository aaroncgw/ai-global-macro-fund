"""
Geopolitical Analyst Agent

This agent analyzes geopolitical news and events to score ETFs based on
geopolitical risks and opportunities.
"""

from src.agents.base_agent import BaseAgent
import json
import logging

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
        Create a prompt for geopolitical analysis.
        
        Args:
            news_data: List of news articles
            universe: List of ETFs to analyze
            
        Returns:
            Formatted prompt string
        """
        # Format news data
        news_summary = self._format_news_data(news_data)
        
        prompt = f"""
        As a geopolitical analyst, assess the following news and events and score each ETF from -1 (strong sell) to 1 (strong buy):
        
        GEOPOLITICAL NEWS AND EVENTS:
        {news_summary}
        
        ANALYSIS FRAMEWORK:
        1. Regional Risks: Which countries/regions face heightened risks?
        2. Trade Relations: Impact on international trade and supply chains
        3. Currency Wars: Effects on currency ETFs (UUP, FXE, FXY, etc.)
        4. Commodity Shocks: Impact on commodity ETFs (GLD, SLV, USO, etc.)
        5. Political Stability: Effects on country-specific ETFs (EWJ, EWG, FXI, etc.)
        6. Global Tensions: Impact on safe-haven assets vs risk assets
        
        SCORING CRITERIA:
        - Score each ETF from -1.0 (strong sell) to 1.0 (strong buy)
        - Consider geopolitical risks, opportunities, and regional impacts
        - Focus on how news affects specific countries, currencies, and commodities
        - Assess both immediate and longer-term geopolitical implications
        
        ETFs TO SCORE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores:
        {{"SPY": 0.1, "EWJ": -0.2, "GLD": 0.3, ...}}
        """
        return prompt
    
    def _format_news_data(self, news_data: list) -> str:
        """Format news data for the prompt."""
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

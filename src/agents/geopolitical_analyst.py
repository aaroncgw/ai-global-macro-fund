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
        Analyze geopolitical news and score ETFs with confidence and reasoning.
        Now processes ETFs one by one for more consistent scoring.
        
        Args:
            state: LangGraph state dictionary containing:
                - news: Geopolitical news data
                - universe: List of ETFs to analyze
                
        Returns:
            Updated state dictionary with geopolitical analyst scores, confidence, and reasoning
        """
        try:
            # Extract data from state
            news = state.get('news', [])
            universe = state.get('universe', [])
            
            # Process each ETF individually for more consistent scoring
            geo_scores = {}
            
            for etf in universe:
                try:
                    logger.info(f"Processing {etf} for geopolitical analysis...")
                    
                    # Create individual ETF analysis prompt
                    prompt = self._create_individual_etf_geo_prompt(news, etf)
                    
                    # Get LLM response with JSON format
                    response = self.llm(prompt, response_format='json_object')
                    
                    # Parse the structured response for this single ETF
                    etf_score = self._parse_single_etf_score(response, etf)
                    geo_scores[etf] = etf_score
                    
                    logger.debug(f"Completed geopolitical analysis for {etf}: score={etf_score.get('score', 0.0)}")
                    
                except Exception as etf_error:
                    logger.error(f"Failed to analyze {etf}: {etf_error}")
                    # Provide neutral score for failed ETF
                    geo_scores[etf] = {
                        'score': 0.0,
                        'confidence': 0.0,
                        'reason': f'Analysis failed for {etf}: {str(etf_error)}'
                    }
            
            # Store geo scores in state
            state['geo_scores'] = geo_scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['geopolitical_analyst'] = {
                'geo_scores': geo_scores,
                'reasoning': f"Geopolitical analysis completed for {len(universe)} ETFs (processed individually)",
                'key_factors': [article.get('title', 'Unknown')[:50] for article in news[:3]] if news else ['No news data available'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Geopolitical analysis completed for {len(universe)} ETFs (individual processing)")
            return state
            
        except Exception as e:
            logger.error(f"Geopolitical analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: {'score': 0.0, 'confidence': 0.0, 'reason': 'Analysis failed due to error'} for etf in state.get('universe', [])}
            state['geo_scores'] = neutral_scores
            return state
    
    
    
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
    
    
    
    def _create_individual_etf_geo_prompt(self, news_data: list, etf: str) -> str:
        """
        Create a prompt for individual ETF geopolitical analysis.
        
        Args:
            news_data: List of news articles
            etf: Single ETF symbol to analyze
            
        Returns:
            Formatted prompt string for single ETF analysis
        """
        # Format news data for this specific ETF analysis
        news_summary = self._format_news_data_comprehensive(news_data)
        
        prompt = f"""
        As a geopolitical analyst, analyze the following comprehensive news from the past 30 days and score ONLY the ETF {etf} from -1 (strong sell) to 1 (strong buy).
        
        You have access to {len(news_data)} news articles covering major global events. Synthesize the key themes and their specific impact on {etf}.
        
        COMPREHENSIVE GEOPOLITICAL NEWS (Last 30 Days):
        {news_summary}
        
        ANALYSIS FRAMEWORK - Identify Major Themes for {etf}:
        1. Regional Risks: How do current geopolitical risks affect {etf} specifically?
           - Conflicts, political instability, regime changes relevant to {etf}
           - Regulatory crackdowns and policy shifts affecting {etf}
           
        2. Trade & Supply Chains: 
           - Trade tensions, tariffs, sanctions affecting {etf}
           - Supply chain disruptions impacting {etf}
           - Reshoring vs offshoring trends affecting {etf}
           
        3. Monetary Policy Divergence:
           - Central bank policy differences affecting {etf}
           - Currency volatility and competitive devaluations affecting {etf}
           - Impact on {etf} if it's currency-focused
           
        4. Commodity & Energy:
           - Oil/gas supply disruptions affecting {etf}
           - Strategic resource competition affecting {etf}
           - Impact on {etf} if it's commodity-focused
           
        5. Regional Economic Outlook:
           - Regional economic conditions affecting {etf}
           - Country-specific risks for {etf}
           - Regional policy changes affecting {etf}
           
        6. Safe Haven Flows:
           - Flight to safety vs risk-on sentiment affecting {etf}
           - Safe haven characteristics of {etf}
           - Risk-off flows affecting {etf}
        
        HISTORICAL CONTEXT FOR {etf} ANALYSIS:
        Consider how current events relate to major geopolitical patterns from the past 25 years and their historical impact on {etf}:
        - 2000-2002: Dot-com crash, 9/11 attacks, Afghanistan War
        - 2003-2007: Iraq War, emerging market growth, commodity supercycle
        - 2008-2009: Global Financial Crisis, quantitative easing era begins
        - 2010-2015: European debt crisis, Arab Spring, China's rise
        - 2016-2020: Brexit, US-China trade war, COVID-19 pandemic
        - 2021-2025: Post-COVID recovery, Russia-Ukraine war, inflation surge, deglobalization
        
        ANALYSIS FRAMEWORK & REQUIREMENTS FOR {etf}:
        - Synthesize ALL {len(news_data)} articles to identify geopolitical themes relevant to {etf}
        - Compare current events to historical precedents and their impact on {etf}
        - Use 25-year historical perspective to assess if current risks are cyclical or structural for {etf}
        - Consider both immediate shocks and longer-term structural shifts affecting {etf}
        
        For {etf}, provide:
        - Score from -1.0 (strong sell) to 1.0 (strong buy) based on:
          * Severity and persistence of geopolitical risks affecting {etf} (vs historical norms)
          * Regional exposure to conflicts/crises affecting {etf} (learning from past crises)
          * Safe haven characteristics of {etf} (how it performed in past crises)
          * Currency and commodity exposure of {etf} (historical volatility patterns)
        - Confidence level from 0.0 (no confidence) to 1.0 (high confidence)
        - Detailed reasoning based on geopolitical factors above
        - Focus specifically on how current geopolitical environment affects {etf}
        
        CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
        {{"{etf}": {{"score": -1 to 1, "confidence": 0-1, "reason": "detailed explanation"}}}}
        
        Do not include any text before or after the JSON. Return only the JSON object.
        """
        return prompt
    
    def _parse_single_etf_score(self, response: str, etf: str) -> dict:
        """
        Parse structured ETF score for a single ETF from LLM response.
        
        Args:
            response: LLM response string
            etf: ETF symbol that was analyzed
            
        Returns:
            Dictionary with ETF score, confidence, and reasoning
        """
        try:
            # Clean the response first
            response = response.strip()
            
            # Try multiple JSON extraction strategies
            json_str = None
            
            # Strategy 1: Look for JSON object boundaries
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
            
            if json_str:
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = json_str.replace('  ', ' ')  # Remove double spaces
                json_str = json_str.replace('\t', ' ')  # Remove tabs
                
                try:
                    scores = json.loads(json_str)
                    
                    # Validate and structure score for this ETF
                    if etf in scores and isinstance(scores[etf], dict):
                        etf_data = scores[etf]
                        score = float(etf_data.get('score', 0.0))
                        confidence = float(etf_data.get('confidence', 0.5))
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Clamp values to valid ranges
                        score = max(-1.0, min(1.0, score))
                        confidence = max(0.0, min(1.0, confidence))
                        
                        return {
                            'score': score,
                            'confidence': confidence,
                            'reason': reason
                        }
                    else:
                        # ETF not found in response
                        return {
                            'score': 0.0,
                            'confidence': 0.0,
                            'reason': f'ETF {etf} not found in LLM response'
                        }
                        
                except json.JSONDecodeError as json_err:
                    logger.warning(f"JSON parse failed for {etf}: {json_err}")
                    # Try fallback extraction
                    return self._extract_single_etf_score_fallback(response, etf)
            else:
                logger.warning(f"No JSON found in LLM response for {etf}")
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'reason': f'No JSON response for {etf}'
                }
                
        except Exception as e:
            logger.error(f"Failed to parse score for {etf}: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'reason': f'Parse error for {etf}: {str(e)}'
            }
    
    def _extract_single_etf_score_fallback(self, text: str, etf: str) -> dict:
        """
        Fallback method to extract score for a single ETF using text patterns.
        
        Args:
            text: Text containing ETF score
            etf: ETF symbol to extract score for
            
        Returns:
            Dictionary with extracted score
        """
        import re
        
        # Look for patterns like "SPY": {"score": 0.5, "confidence": 0.8, "reason": "..."}
        pattern = f'"{etf}"\\s*:\\s*\\{{[^}}]*"score"\\s*:\\s*([-+]?\\d*\\.?\\d+)[^}}]*"confidence"\\s*:\\s*([-+]?\\d*\\.?\\d+)[^}}]*"reason"\\s*:\\s*"([^"]*)"'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            try:
                score = float(match.group(1))
                confidence = float(match.group(2))
                reason = match.group(3)
                
                # Clamp values
                score = max(-1.0, min(1.0, score))
                confidence = max(0.0, min(1.0, confidence))
                
                return {
                    'score': score,
                    'confidence': confidence,
                    'reason': reason
                }
            except (ValueError, IndexError):
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'reason': f'Failed to extract score for {etf} from text pattern'
                }
        else:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'reason': f'No score pattern found for {etf} in response'
            }


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
        print(f"  Geo scores: {result_state.get('geo_scores', {})}")
        
        # Show detailed output for first ETF
        geo_scores = result_state.get('geo_scores', {})
        if geo_scores:
            first_etf = list(geo_scores.keys())[0]
            etf_data = geo_scores[first_etf]
            print(f"  Sample ETF ({first_etf}):")
            print(f"    Score: {etf_data.get('score', 0.0)}")
            print(f"    Confidence: {etf_data.get('confidence', 0.0)}")
            print(f"    Reason: {etf_data.get('reason', 'No reason')[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Geopolitical analyst test completed!")

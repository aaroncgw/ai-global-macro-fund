"""
Risk Manager Agent for ETF Risk Assessment

This agent assesses risks and sets position limits based on combined analyst scores,
macro data, and ETF data. Mimics the original ai-hedge-fund RiskManager but for batch ETFs.
"""

from src.agents.base_agent import BaseAgent
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager(BaseAgent):
    """
    Risk Manager Agent that assesses risks and sets position limits for ETFs.
    
    This agent focuses on:
    - Risk assessment based on combined analyst scores
    - Position sizing and risk limits
    - Volatility and correlation analysis
    - Macro risk factors
    - Portfolio risk management
    """
    
    def __init__(self, agent_name: str = "RiskManager"):
        """Initialize the risk manager agent."""
        super().__init__(agent_name)
        self.specialization = "risk_management"
        self.analysis_focus = "risk_assessment"
        self.role = "risk_manager"
    
    def analyze(self, state: dict) -> dict:
        """
        Analyze risks and adjust scores for ETFs (required by BaseAgent).
        
        Args:
            state: LangGraph state dictionary
            
        Returns:
            Updated state dictionary with risk assessments
        """
        return self.assess(state)
    
    def assess(self, state: dict) -> dict:
        """
        Assess risks for ETFs using macro data, ETF data, and news for text-based risk understanding.
        
        Args:
            state: LangGraph state dictionary containing:
                - macro_data: Macro economic indicators
                - etf_data: ETF price data
                - news: Geopolitical and financial news
                - universe: List of ETFs to assess
                
        Returns:
            Updated state dictionary with risk metrics
        """
        try:
            # Extract data from state
            macro_data = state.get('macro_data', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            news = state.get('news', [])
            universe = state.get('universe', [])
            
            # Create risk assessment prompt with news input
            prompt = f"""
            Independent of analyst scores, assess risks for each ETF in {universe} using macro data, ETF data, and geopolitical/financial news for text-based insights.
            
            MACRO ECONOMIC DATA:
            {self._format_macro_data(macro_data)}
            
            ETF PERFORMANCE DATA:
            {self._format_etf_performance(etf_data, universe)}
            
            GEOPOLITICAL AND FINANCIAL NEWS:
            {self._format_news_for_risk_assessment(news)}
            
            RISK ASSESSMENT REQUIREMENTS:
            1. For each ETF, determine risk level: "low", "medium", or "high"
            2. Calculate volatility as standard deviation of returns (use historical data if available)
            3. Provide detailed reasoning for risk assessment based on macro factors, news impact, and volatility
            
            RISK LEVEL CRITERIA:
            - LOW: Stable, diversified, low volatility, strong fundamentals
            - MEDIUM: Moderate volatility, some concentration risk, mixed fundamentals
            - HIGH: High volatility, concentrated exposure, weak fundamentals, high correlation
            
            MACRO RISK FACTORS TO CONSIDER:
            - Inflation impact on bonds vs commodities
            - Interest rate sensitivity
            - Currency volatility and competitive devaluations
            - Regional economic stability
            - Trade tensions and geopolitical conflicts
            - Central bank policy differences
            - Commodity supply disruptions
            - Safe haven flows during uncertainty
            
            NEWS-BASED RISK FACTORS:
            - Geopolitical tensions affecting specific regions
            - Trade wars and tariff impacts
            - Currency wars and competitive devaluations
            - Central bank policy changes
            - Commodity supply disruptions
            - Political instability
            - Economic sanctions
            - Market sentiment shifts
            
            PRICE-RELATED RISK FACTORS:
            - Historical volatility patterns and trends
            - Price momentum and technical indicators
            - Volume patterns and liquidity risks
            - Drawdown analysis and maximum loss potential
            - Correlation with market indices and other assets
            - Price impact of large trades
            
            Output dict format:
            {{"ETF": {{"risk_level": "low/medium/high", "volatility": float, "reason": "detailed explanation"}}}}
            """
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse the structured response
            risk_metrics = self._parse_risk_metrics(response, universe)
            
            # Store risk metrics in state
            state['risk_metrics'] = risk_metrics
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['risk_manager'] = {
                'risk_metrics': risk_metrics,
                'reasoning': f"Independent risk assessment based on macro data, ETF data, and {len(news)} news articles for {len(universe)} ETFs",
                'key_factors': ['macro_risks', 'geopolitical_risks', 'volatility', 'news_impact'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Risk assessment completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return neutral risk metrics on error
            neutral_metrics = {etf: {'risk_level': 'medium', 'volatility': 0.0, 'reason': 'Risk assessment failed due to error'} for etf in state.get('universe', [])}
            state['risk_metrics'] = neutral_metrics
            return state
    
    def _format_news_for_risk_assessment(self, news: list) -> str:
        """Format news data for risk assessment prompt."""
        if not news:
            return "No news data available"
        
        formatted = []
        for i, article in enumerate(news[:10], 1):  # Limit to top 10 articles
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary')
            sentiment = article.get('sentiment', 'neutral')
            impact = article.get('impact', 'medium')
            
            formatted.append(f"{i}. {title}")
            formatted.append(f"   Summary: {summary[:200]}...")
            formatted.append(f"   Sentiment: {sentiment}, Impact: {impact}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_combined_scores(self, combined_scores: dict) -> str:
        """Format combined analyst scores for the prompt."""
        if not combined_scores:
            return "No combined scores available"
        
        formatted = []
        for etf, score in combined_scores.items():
            formatted.append(f"  {etf}: {score:.3f}")
        
        return '\n'.join(formatted) if formatted else "No combined scores available"
    
    def _format_macro_data(self, macro_data: dict) -> str:
        """Format macro data for the prompt."""
        if not macro_data:
            return "No macro data available"
        
        formatted = []
        for indicator, data in macro_data.items():
            if 'error' not in data and data.get('latest_value') is not None:
                latest_value = data.get('latest_value', 'N/A')
                trend = data.get('trend', 'unknown')
                formatted.append(f"- {indicator}: {latest_value} (trend: {trend})")
            else:
                formatted.append(f"- {indicator}: Error - {data.get('error', 'No data')}")
        
        return '\n'.join(formatted) if formatted else "No macro data available"
    
    def _format_etf_performance(self, etf_data: pd.DataFrame, universe: list) -> str:
        """Format ETF performance data for the prompt."""
        if etf_data.empty:
            return "No ETF performance data available"
        
        formatted = []
        for etf in universe:
            if etf in etf_data.columns:
                returns = etf_data[etf].pct_change().dropna()
                if not returns.empty:
                    try:
                        # Calculate volatility (annualized)
                        volatility = returns.std() * (252 ** 0.5)
                        # Calculate recent performance
                        recent_return = returns.tail(5).mean() * 252  # Annualized
                        # Calculate max drawdown
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min()
                        
                        formatted.append(f"- {etf}: Vol={volatility:.3f}, Recent={recent_return:.3f}, MaxDD={max_drawdown:.3f}")
                    except Exception as e:
                        formatted.append(f"- {etf}: Error calculating metrics - {str(e)}")
                else:
                    formatted.append(f"- {etf}: No returns data")
            else:
                formatted.append(f"- {etf}: Not in data")
        
        return '\n'.join(formatted) if formatted else "No ETF performance data available"
    
    def _parse_risk_assessments(self, response: str, universe: list) -> dict:
        """
        Parse structured risk assessments from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to assess
            
        Returns:
            Dictionary with risk assessments
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                assessments = json.loads(json_str)
                
                # Validate and structure assessments
                validated_assessments = {}
                for etf in universe:
                    if etf in assessments and isinstance(assessments[etf], dict):
                        etf_data = assessments[etf]
                        risk_level = str(etf_data.get('risk_level', 'medium')).lower()
                        adjusted_score = float(etf_data.get('adjusted_score', 0.0))
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Validate risk level
                        if risk_level not in ['low', 'medium', 'high']:
                            risk_level = 'medium'
                        
                        # Clamp adjusted score
                        adjusted_score = max(-1.0, min(1.0, adjusted_score))
                        
                        validated_assessments[etf] = {
                            'risk_level': risk_level,
                            'adjusted_score': adjusted_score,
                            'reason': reason
                        }
                    else:
                        # Default values for missing ETFs
                        validated_assessments[etf] = {
                            'risk_level': 'medium',
                            'adjusted_score': 0.0,
                            'reason': 'No risk assessment available'
                        }
                
                return validated_assessments
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: {'risk_level': 'medium', 'adjusted_score': 0.0, 'reason': 'No JSON response'} for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse risk assessments from response: {e}")
            return {etf: {'risk_level': 'medium', 'adjusted_score': 0.0, 'reason': f'Parse error: {str(e)}'} for etf in universe}
    
    def _parse_risk_metrics(self, response: str, universe: list) -> dict:
        """
        Parse structured risk metrics from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to assess
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                metrics = json.loads(json_str)
                
                # Validate and structure metrics
                validated_metrics = {}
                for etf in universe:
                    if etf in metrics and isinstance(metrics[etf], dict):
                        etf_data = metrics[etf]
                        risk_level = str(etf_data.get('risk_level', 'medium')).lower()
                        volatility = float(etf_data.get('volatility', 0.0))
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Validate risk level
                        if risk_level not in ['low', 'medium', 'high']:
                            risk_level = 'medium'
                        
                        # Ensure volatility is non-negative
                        volatility = max(0.0, volatility)
                        
                        validated_metrics[etf] = {
                            'risk_level': risk_level,
                            'volatility': volatility,
                            'reason': reason
                        }
                    else:
                        # Default values for missing ETFs
                        validated_metrics[etf] = {
                            'risk_level': 'medium',
                            'volatility': 0.0,
                            'reason': 'No risk assessment available'
                        }
                
                return validated_metrics
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: {'risk_level': 'medium', 'volatility': 0.0, 'reason': 'No JSON response'} for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse risk metrics from response: {e}")
            return {etf: {'risk_level': 'medium', 'volatility': 0.0, 'reason': f'Parse error: {str(e)}'} for etf in universe}


# Example usage and testing
if __name__ == "__main__":
    print("Risk Manager Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = RiskManager("TestRiskManager")
        print(f"✓ Risk manager agent initialized")
        print(f"  Specialization: {agent.specialization}")
        print(f"  Provider: {agent.get_provider_info()['provider']}")
        
        # Test with sample state (independent risk assessment)
        sample_state = {
            'macro_data': {
                'CPIAUCSL': {'latest_value': 300.0, 'trend': 'increasing'},
                'UNRATE': {'latest_value': 3.5, 'trend': 'stable'},
                'FEDFUNDS': {'latest_value': 5.25, 'trend': 'increasing'}
            },
            'etf_data': pd.DataFrame({
                'SPY': [100, 101, 102, 103, 104],
                'TLT': [150, 149, 148, 147, 146],
                'GLD': [180, 181, 182, 183, 184]
            }),
            'news': [
                {
                    'title': 'Federal Reserve Maintains Hawkish Stance on Interest Rates',
                    'summary': 'Fed signals continued rate hikes to combat inflation, affecting bond markets',
                    'sentiment': 'negative',
                    'impact': 'high'
                },
                {
                    'title': 'Geopolitical Tensions Rise in Middle East',
                    'summary': 'Regional conflicts escalate, driving safe-haven demand for gold',
                    'sentiment': 'negative',
                    'impact': 'high'
                }
            ],
            'universe': ['SPY', 'TLT', 'GLD']
        }
        
        # Test independent risk assessment
        result_state = agent.assess(sample_state)
        print(f"✓ Independent risk assessment completed")
        print(f"  Risk metrics: {result_state.get('risk_metrics', {})}")
        
        # Show detailed output for first ETF
        risk_metrics = result_state.get('risk_metrics', {})
        if risk_metrics:
            first_etf = list(risk_metrics.keys())[0]
            etf_data = risk_metrics[first_etf]
            print(f"  Sample ETF ({first_etf}):")
            print(f"    Risk Level: {etf_data.get('risk_level', 'unknown')}")
            print(f"    Volatility: {etf_data.get('volatility', 0.0):.3f}")
            print(f"    Reason: {etf_data.get('reason', 'No reason')[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Risk manager test completed!")

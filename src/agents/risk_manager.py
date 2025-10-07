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
        Now processes ETFs one by one for more consistent risk assessment.
        
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
            
            # Calculate actual volatility for each ETF
            etf_volatilities = {}
            if not etf_data.empty:
                for etf in universe:
                    if etf in etf_data.columns:
                        returns = etf_data[etf].pct_change().dropna()
                        if len(returns) > 0:
                            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                            # Ensure volatility is a float, not a pandas Series
                            if hasattr(volatility, 'iloc'):
                                volatility = float(volatility.iloc[0]) if len(volatility) > 0 else 0.0
                            else:
                                volatility = float(volatility)
                            etf_volatilities[etf] = round(volatility, 4)
                        else:
                            etf_volatilities[etf] = 0.0
                    else:
                        etf_volatilities[etf] = 0.0
            else:
                etf_volatilities = {etf: 0.0 for etf in universe}
            
            # Process each ETF individually for more consistent risk assessment
            risk_metrics = {}
            
            for etf in universe:
                try:
                    logger.info(f"Processing {etf} for risk assessment...")
                    
                    # Create individual ETF risk assessment prompt
                    prompt = self._create_individual_etf_risk_prompt(macro_data, etf_data, news, etf, etf_volatilities.get(etf, 0.0))
                    
                    # Get LLM response with JSON format
                    try:
                        response = self.llm(prompt, response_format='json_object')
                    except Exception as e:
                        logger.error(f"LLM call failed for {etf}: {e}")
                        # Provide neutral risk metrics for failed ETF
                        risk_metrics[etf] = {
                            'risk_level': 'medium',
                            'volatility': etf_volatilities.get(etf, 0.0),
                            'reason': f'LLM call failed for {etf}: {str(e)}'
                        }
                        continue
                    
                    # Parse the structured response for this single ETF
                    etf_risk = self._parse_single_etf_risk(response, etf, etf_volatilities.get(etf, 0.0))
                    risk_metrics[etf] = etf_risk
                    
                    logger.debug(f"Completed risk assessment for {etf}: risk_level={etf_risk.get('risk_level', 'unknown')}")
                    
                except Exception as etf_error:
                    logger.error(f"Failed to assess risk for {etf}: {etf_error}")
                    # Provide neutral risk metrics for failed ETF
                    risk_metrics[etf] = {
                        'risk_level': 'medium',
                        'volatility': etf_volatilities.get(etf, 0.0),
                        'reason': f'Risk assessment failed for {etf}: {str(etf_error)}'
                    }
            
            # Store risk metrics in state
            state['risk_metrics'] = risk_metrics
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['risk_manager'] = {
                'risk_metrics': risk_metrics,
                'reasoning': f"Independent risk assessment completed for {len(universe)} ETFs (processed individually)",
                'key_factors': ['macro_risks', 'geopolitical_risks', 'volatility', 'news_impact'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Risk assessment completed for {len(universe)} ETFs (individual processing)")
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
    
    def _parse_risk_metrics(self, response: str, universe: list, etf_volatilities: dict) -> dict:
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
                        # Use calculated volatility instead of parsing from LLM
                        volatility = etf_volatilities.get(etf, 0.0)
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Validate risk level
                        if risk_level not in ['low', 'medium', 'high']:
                            risk_level = 'medium'
                        
                        validated_metrics[etf] = {
                            'risk_level': risk_level,
                            'volatility': volatility,
                            'reason': reason
                        }
                    else:
                        # Default values for missing ETFs
                        validated_metrics[etf] = {
                            'risk_level': 'medium',
                            'volatility': etf_volatilities.get(etf, 0.0),
                            'reason': 'No risk assessment available'
                        }
                
                return validated_metrics
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: {'risk_level': 'medium', 'volatility': etf_volatilities.get(etf, 0.0), 'reason': 'No JSON response'} for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse risk metrics from response: {e}")
            return {etf: {'risk_level': 'medium', 'volatility': etf_volatilities.get(etf, 0.0), 'reason': f'Parse error: {str(e)}'} for etf in universe}
    
    def _create_individual_etf_risk_prompt(self, macro_data: dict, etf_data: pd.DataFrame, news: list, etf: str, volatility: float) -> str:
        """
        Create a prompt for individual ETF risk assessment.
        
        Args:
            macro_data: Macro economic indicators
            etf_data: ETF price data
            news: Geopolitical and financial news
            etf: Single ETF symbol to assess
            volatility: Calculated volatility for this ETF
            
        Returns:
            Formatted prompt string for single ETF risk assessment
        """
        # Format data for this specific ETF
        macro_summary = self._format_macro_data(macro_data)
        etf_performance = self._format_individual_etf_performance(etf_data, etf)
        news_summary = self._format_news_for_risk_assessment(news)
        
        prompt = f"""
        Independent of analyst scores, assess risks for ONLY the ETF {etf} using macro data, ETF data, and geopolitical/financial news for text-based insights.
        
        MACRO ECONOMIC DATA:
        {macro_summary}
        
        {etf} PERFORMANCE DATA:
        {etf_performance}
        
        CALCULATED VOLATILITY (Annualized) for {etf}: {volatility:.4f}
        
        GEOPOLITICAL AND FINANCIAL NEWS:
        {news_summary}
        
        RISK ASSESSMENT REQUIREMENTS FOR {etf}:
        1. For {etf}, determine risk level: "low", "medium", or "high"
        2. Use the provided volatility calculation ({volatility:.4f}) for {etf}
        3. Provide detailed reasoning for risk assessment based on macro factors, news impact, and volatility
        
        RISK LEVEL CRITERIA FOR {etf}:
        - LOW: {etf} is stable, diversified, low volatility, strong fundamentals
        - MEDIUM: {etf} has moderate volatility, some concentration risk, mixed fundamentals
        - HIGH: {etf} has high volatility, concentrated exposure, weak fundamentals, high correlation
        
        MACRO RISK FACTORS TO CONSIDER FOR {etf}:
        - Inflation impact on {etf} based on its asset class
        - Interest rate sensitivity of {etf}
        - Currency volatility affecting {etf}
        - Regional economic stability affecting {etf}
        - Trade tensions and geopolitical conflicts affecting {etf}
        - Central bank policy differences affecting {etf}
        - Commodity supply disruptions affecting {etf}
        - Safe haven flows affecting {etf}
        
        NEWS-BASED RISK FACTORS FOR {etf}:
        - Geopolitical tensions affecting {etf} specifically
        - Trade wars and tariff impacts on {etf}
        - Currency wars affecting {etf}
        - Central bank policy changes affecting {etf}
        - Commodity supply disruptions affecting {etf}
        - Political instability affecting {etf}
        - Economic sanctions affecting {etf}
        - Market sentiment shifts affecting {etf}
        
        PRICE-RELATED RISK FACTORS FOR {etf}:
        - Historical volatility patterns for {etf} (current: {volatility:.4f})
        - Price momentum and technical indicators for {etf}
        - Volume patterns and liquidity risks for {etf}
        - Drawdown analysis for {etf}
        - Correlation with market indices and other assets
        - Price impact of large trades on {etf}
        
        EXAMPLES OF EXPECTED OUTPUT:
        Example 1 - High volatility ETF with geopolitical risks:
        {{"{etf}": {{"risk_level": "high", "volatility": {volatility:.4f}, "reason": "Elevated volatility ({volatility:.1%}) combined with trade war tensions and policy uncertainty creates high risk profile for {etf}"}}}}
        
        Example 2 - Stable market environment:
        {{"{etf}": {{"risk_level": "low", "volatility": {volatility:.4f}, "reason": "Low volatility ({volatility:.1%}) with stable economic growth and accommodative policy supports {etf}"}}}}
        
        Example 3 - Crisis period:
        {{"{etf}": {{"risk_level": "high", "volatility": {volatility:.4f}, "reason": "Crisis-level volatility ({volatility:.1%}) with multiple risk factors: recession fears, geopolitical tensions, and credit stress affecting {etf}"}}}}
        
        CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
        {{"{etf}": {{"risk_level": "low/medium/high", "volatility": {volatility:.4f}, "reason": "detailed explanation"}}}}
        
        Do not include any text before or after the JSON. Return only the JSON object.
        """
        return prompt
    
    def _format_individual_etf_performance(self, etf_data: pd.DataFrame, etf: str) -> str:
        """Format individual ETF performance data for the prompt."""
        if etf_data.empty or etf not in etf_data.columns:
            return f"No {etf} performance data available"
        
        returns = etf_data[etf].pct_change().dropna()
        if returns.empty:
            return f"No {etf} returns data available"
        
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
            
            return f"- {etf}: Vol={volatility:.3f}, Recent={recent_return:.3f}, MaxDD={max_drawdown:.3f}"
        except Exception as e:
            return f"- {etf}: Error calculating metrics - {str(e)}"
    
    def _parse_single_etf_risk(self, response: str, etf: str, volatility: float) -> dict:
        """
        Parse structured risk metrics for a single ETF from LLM response.
        
        Args:
            response: LLM response string
            etf: ETF symbol that was assessed
            volatility: Calculated volatility for this ETF
            
        Returns:
            Dictionary with risk metrics
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
                    metrics = json.loads(json_str)
                    
                    # Validate and structure metrics for this ETF
                    if etf in metrics and isinstance(metrics[etf], dict):
                        etf_data = metrics[etf]
                        risk_level = str(etf_data.get('risk_level', 'medium')).lower()
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Validate risk level
                        if risk_level not in ['low', 'medium', 'high']:
                            risk_level = 'medium'
                        
                        return {
                            'risk_level': risk_level,
                            'volatility': volatility,  # Use calculated volatility
                            'reason': reason
                        }
                    else:
                        # ETF not found in response
                        return {
                            'risk_level': 'medium',
                            'volatility': volatility,
                            'reason': f'ETF {etf} not found in LLM response'
                        }
                        
                except json.JSONDecodeError as json_err:
                    logger.warning(f"JSON parse failed for {etf}: {json_err}")
                    # Try fallback extraction
                    return self._extract_single_etf_risk_fallback(response, etf, volatility)
            else:
                logger.warning(f"No JSON found in LLM response for {etf}")
                return {
                    'risk_level': 'medium',
                    'volatility': volatility,
                    'reason': f'No JSON response for {etf}'
                }
                
        except Exception as e:
            logger.error(f"Failed to parse risk metrics for {etf}: {e}")
            return {
                'risk_level': 'medium',
                'volatility': volatility,
                'reason': f'Parse error for {etf}: {str(e)}'
            }
    
    def _extract_single_etf_risk_fallback(self, text: str, etf: str, volatility: float) -> dict:
        """
        Fallback method to extract risk metrics for a single ETF using text patterns.
        
        Args:
            text: Text containing ETF risk metrics
            etf: ETF symbol to extract risk metrics for
            volatility: Calculated volatility for this ETF
            
        Returns:
            Dictionary with extracted risk metrics
        """
        import re
        
        # Look for patterns like "SPY": {"risk_level": "high", "volatility": 0.25, "reason": "..."}
        pattern = f'"{etf}"\\s*:\\s*\\{{[^}}]*"risk_level"\\s*:\\s*"([^"]*)"[^}}]*"reason"\\s*:\\s*"([^"]*)"'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            try:
                risk_level = match.group(1).lower()
                reason = match.group(2)
                
                # Validate risk level
                if risk_level not in ['low', 'medium', 'high']:
                    risk_level = 'medium'
                
                return {
                    'risk_level': risk_level,
                    'volatility': volatility,  # Use calculated volatility
                    'reason': reason
                }
            except (ValueError, IndexError):
                return {
                    'risk_level': 'medium',
                    'volatility': volatility,
                    'reason': f'Failed to extract risk metrics for {etf} from text pattern'
                }
        else:
            return {
                'risk_level': 'medium',
                'volatility': volatility,
                'reason': f'No risk pattern found for {etf} in response'
            }


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

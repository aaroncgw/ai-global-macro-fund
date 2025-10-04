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
    
    def assess(self, state: dict) -> dict:
        """
        Assess risks and adjust scores for ETFs based on combined analyst scores.
        
        Args:
            state: LangGraph state dictionary containing:
                - macro_scores: Macro economist scores
                - geo_scores: Geopolitical analyst scores
                - macro_data: Macro economic indicators
                - etf_data: ETF price data
                - universe: List of ETFs to assess
                
        Returns:
            Updated state dictionary with risk assessments
        """
        try:
            # Extract data from state
            macro_scores = state.get('macro_scores', {})
            geo_scores = state.get('geo_scores', {})
            macro_data = state.get('macro_data', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            # Combine analyst scores (simple average for now)
            combined = {}
            for etf in universe:
                macro_score = macro_scores.get(etf, {'score': 0}).get('score', 0)
                geo_score = geo_scores.get(etf, {'score': 0}).get('score', 0)
                combined[etf] = (macro_score + geo_score) / 2
            
            # Create risk assessment prompt
            prompt = f"""
            For each ETF, assess risks on combined scores {combined}, macro_data {macro_data}, etf_data {etf_data.pct_change().tail(252) if not etf_data.empty else 'No ETF data available'}.
            
            COMBINED ANALYST SCORES:
            {self._format_combined_scores(combined)}
            
            MACRO ECONOMIC DATA:
            {self._format_macro_data(macro_data)}
            
            ETF PERFORMANCE DATA:
            {self._format_etf_performance(etf_data, universe)}
            
            RISK ASSESSMENT REQUIREMENTS:
            1. For each ETF, determine risk level: "low", "medium", or "high"
            2. Adjust the combined score based on risk factors
            3. Provide detailed reasoning for risk assessment
            4. Consider volatility, correlation, macro risks, and position sizing
            
            RISK FACTORS TO CONSIDER:
            - Volatility and price stability
            - Correlation with other assets
            - Macro economic sensitivity
            - Geopolitical risk exposure
            - Liquidity and market depth
            - Concentration risk
            - Drawdown potential
            
            RISK LEVEL CRITERIA:
            - LOW: Stable, diversified, low volatility, strong fundamentals
            - MEDIUM: Moderate volatility, some concentration risk, mixed fundamentals
            - HIGH: High volatility, concentrated exposure, weak fundamentals, high correlation
            
            SCORE ADJUSTMENT RULES:
            - Low risk: Maintain or slightly boost score
            - Medium risk: Moderate score reduction (0.1-0.3 points)
            - High risk: Significant score reduction (0.3-0.5 points)
            
            Output dict format:
            {{"ETF": {{"risk_level": "low/medium/high", "adjusted_score": -1 to 1, "reason": "detailed explanation"}}}}
            """
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse the structured response
            risk_assessments = self._parse_risk_assessments(response, universe)
            
            # Store risk assessments in state
            state['risk_assessments'] = risk_assessments
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['risk_manager'] = {
                'risk_assessments': risk_assessments,
                'reasoning': f"Risk assessment based on combined scores from {len(macro_scores)} macro and {len(geo_scores)} geo analysts",
                'key_factors': ['combined_scores', 'macro_data', 'etf_performance', 'volatility'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Risk assessment completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return neutral risk assessments on error
            neutral_assessments = {etf: {'risk_level': 'medium', 'adjusted_score': 0.0, 'reason': 'Risk assessment failed due to error'} for etf in state.get('universe', [])}
            state['risk_assessments'] = neutral_assessments
            return state
    
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
        
        # Test with sample state
        sample_state = {
            'macro_scores': {
                'SPY': {'score': 0.3, 'confidence': 0.8, 'reason': 'Positive macro trends'},
                'QQQ': {'score': 0.5, 'confidence': 0.7, 'reason': 'Tech growth outlook'},
                'TLT': {'score': -0.2, 'confidence': 0.6, 'reason': 'Rising rates pressure'}
            },
            'geo_scores': {
                'SPY': {'score': 0.1, 'confidence': 0.5, 'reason': 'US stability'},
                'QQQ': {'score': 0.2, 'confidence': 0.6, 'reason': 'Tech resilience'},
                'TLT': {'score': 0.4, 'confidence': 0.7, 'reason': 'Safe haven demand'}
            },
            'macro_data': {
                'CPIAUCSL': {'latest_value': 300.0, 'trend': 'increasing'},
                'UNRATE': {'latest_value': 3.5, 'trend': 'stable'}
            },
            'etf_data': pd.DataFrame({
                'SPY': [100, 101, 102, 103, 104],
                'QQQ': [200, 201, 202, 203, 204],
                'TLT': [150, 149, 148, 147, 146]
            }),
            'universe': ['SPY', 'QQQ', 'TLT']
        }
        
        # Test risk assessment
        result_state = agent.assess(sample_state)
        print(f"✓ Risk assessment completed")
        print(f"  Risk assessments: {result_state.get('risk_assessments', {})}")
        
        # Show detailed output for first ETF
        risk_assessments = result_state.get('risk_assessments', {})
        if risk_assessments:
            first_etf = list(risk_assessments.keys())[0]
            etf_data = risk_assessments[first_etf]
            print(f"  Sample ETF ({first_etf}):")
            print(f"    Risk Level: {etf_data.get('risk_level', 'unknown')}")
            print(f"    Adjusted Score: {etf_data.get('adjusted_score', 0.0)}")
            print(f"    Reason: {etf_data.get('reason', 'No reason')[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Risk manager test completed!")

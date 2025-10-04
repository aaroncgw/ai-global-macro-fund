"""
Macro Economist Agent

This agent analyzes macroeconomic data and scores ETFs based on
macroeconomic trends, inflation impacts, and economic cycles.
"""

from src.agents.base_agent import BaseAgent
from src.config import DEFAULT_HORIZON
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class MacroEconomistAgent(BaseAgent):
    """
    Macro Economist Agent that analyzes macroeconomic data and scores ETFs.
    
    This agent focuses on:
    - Economic indicators (GDP, inflation, unemployment, interest rates)
    - Macro trends and cycles
    - Impact of macro factors on different asset classes
    - Scoring ETFs from -1 (sell) to 1 (buy) based on macro environment
    """
    
    def __init__(self, agent_name: str = "MacroEconomistAgent"):
        """Initialize the macro economist agent."""
        super().__init__(agent_name)
        self.specialization = "macro_economic_analysis"
        self.analysis_focus = "economic_indicators"
    
    def analyze(self, state: dict) -> dict:
        """
        Analyze macroeconomic data and score ETFs with confidence and reasoning.
        
        Args:
            state: LangGraph state dictionary containing:
                - macro_data: Macro economic indicators
                - etf_data: ETF price data
                - universe: List of ETFs to analyze
                
        Returns:
            Updated state dictionary with macro economist scores, confidence, and reasoning
        """
        try:
            # Extract data from state
            macro_data = state.get('macro_data', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            # Create analysis prompt for batch processing
            prompt = f"""
            For each ETF in {universe}, analyze macro data {macro_data} and returns {etf_data.pct_change().tail(252) if not etf_data.empty else 'No ETF data available'}.
            
            MACRO ECONOMIC INDICATORS:
            {self._format_macro_indicators(macro_data)}
            
            ETF RETURNS (252-day historical):
            {self._format_etf_returns(etf_data.pct_change().tail(252) if not etf_data.empty else pd.DataFrame(), universe)}
            
            ANALYSIS REQUIREMENTS:
            1. For each ETF, provide a score from -1 (strong sell) to 1 (strong buy)
            2. Provide confidence level from 0 (no confidence) to 1 (high confidence)
            3. Provide detailed reasoning for each score based on macro factors
            4. Consider inflation impact, interest rates, economic growth, currency strength
            5. Focus on {DEFAULT_HORIZON} horizon
            
            Output dict format:
            {{"ETF": {{"score": -1 to 1, "confidence": 0-1, "reason": "detailed explanation"}}}}
            """
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse the structured response
            macro_scores = self._parse_structured_scores(response, universe)
            
            # Store macro scores in state
            state['macro_scores'] = macro_scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['macro_economist'] = {
                'macro_scores': macro_scores,
                'reasoning': f"Macro economist analysis based on {len(macro_data)} indicators and {len(universe)} ETFs",
                'key_factors': list(macro_data.keys()) if macro_data else ['No macro data available'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Macro economist analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Macro economist analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: {'score': 0.0, 'confidence': 0.0, 'reason': 'Analysis failed due to error'} for etf in state.get('universe', [])}
            state['macro_scores'] = neutral_scores
            return state
    
    def _create_macro_analysis_prompt(self, macro_data: dict, etf_data: pd.DataFrame, universe: list) -> str:
        """
        Create a prompt for macroeconomic analysis.
        
        Args:
            macro_data: Dictionary of macro economic indicators
            etf_data: DataFrame with ETF price data
            universe: List of ETFs to analyze
            
        Returns:
            Formatted prompt string
        """
        # Calculate ETF returns for analysis (use full 25-year dataset)
        etf_returns = etf_data.pct_change().dropna() if etf_data is not None and not etf_data.empty else pd.DataFrame()
        
        # Format macro indicators
        macro_summary = self._format_macro_indicators(macro_data)
        
        # Format ETF returns
        etf_summary = self._format_etf_returns(etf_returns, universe)
        
        prompt = f"""
        As a macro economist, analyze the following data and score each ETF from -1 (strong sell) to 1 (strong buy):
        
        MACRO ECONOMIC DATA:
        {macro_summary}
        
        ETF RETURNS (25-year historical analysis):
        {etf_summary}
        
        ANALYSIS FRAMEWORK:
        1. Inflation Impact: How does current inflation affect bonds (TLT, IEF) vs commodities (GLD, SLV)?
        2. Interest Rate Environment: Impact on bond ETFs vs equity ETFs
        3. Economic Growth: Which regions/sectors benefit from current growth trends?
        4. Currency Strength: Impact on international ETFs (EWJ, EWG, etc.)
        5. Economic Cycle Position: Where are we in the cycle and what assets perform best?
        
        SCORING CRITERIA:
        - Score each ETF from -1.0 (strong sell) to 1.0 (strong buy)
        - Consider macro trends, inflation, rates, growth, and currency factors
        - Focus on {DEFAULT_HORIZON} horizon
        
        ETFs TO SCORE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores:
        {{"SPY": 0.2, "QQQ": 0.1, "TLT": -0.3, ...}}
        """
        return prompt
    
    def _format_macro_indicators(self, macro_data: dict) -> str:
        """Format macro indicators for the prompt."""
        if not macro_data:
            return "No macro indicators available"
        
        formatted = []
        for indicator, data in macro_data.items():
            if 'error' not in data and data.get('latest_value') is not None:
                latest_value = data.get('latest_value', 'N/A')
                periods = data.get('periods', 0)
                formatted.append(f"- {indicator}: {latest_value} ({periods} periods)")
            else:
                formatted.append(f"- {indicator}: Error - {data.get('error', 'No data')}")
        
        return '\n'.join(formatted) if formatted else "No macro indicators available"
    
    def _format_etf_returns(self, etf_returns: pd.DataFrame, universe: list) -> str:
        """Format ETF returns for the prompt."""
        if etf_returns.empty:
            return "No ETF returns data available"
        
        formatted = []
        for etf in universe:
            if etf in etf_returns.columns:
                returns = etf_returns[etf].dropna()
                if not returns.empty:
                    # Safely extract scalar values
                    try:
                        recent_return = returns.iloc[-1]
                        if hasattr(recent_return, 'item'):
                            recent_return = recent_return.item()
                        recent_return = float(recent_return)
                    except (ValueError, TypeError, IndexError):
                        recent_return = 0.0
                    
                    try:
                        avg_return = returns.mean()
                        if hasattr(avg_return, 'item'):
                            avg_return = avg_return.item()
                        avg_return = float(avg_return)
                    except (ValueError, TypeError):
                        avg_return = 0.0
                    
                    formatted.append(f"- {etf}: Recent: {recent_return:.3f}, Avg: {avg_return:.3f}")
                else:
                    formatted.append(f"- {etf}: No returns data")
            else:
                formatted.append(f"- {etf}: Not in data")
        
        return '\n'.join(formatted) if formatted else "No ETF returns data available"
    
    def _parse_structured_scores(self, response: str, universe: list) -> dict:
        """
        Parse structured ETF scores with confidence and reasoning from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to score
            
        Returns:
            Dictionary with ETF scores, confidence, and reasoning
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                scores = json.loads(json_str)
                
                # Validate and structure scores
                validated_scores = {}
                for etf in universe:
                    if etf in scores and isinstance(scores[etf], dict):
                        etf_data = scores[etf]
                        score = float(etf_data.get('score', 0.0))
                        confidence = float(etf_data.get('confidence', 0.5))
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Clamp values to valid ranges
                        score = max(-1.0, min(1.0, score))
                        confidence = max(0.0, min(1.0, confidence))
                        
                        validated_scores[etf] = {
                            'score': score,
                            'confidence': confidence,
                            'reason': reason
                        }
                    else:
                        # Default values for missing ETFs
                        validated_scores[etf] = {
                            'score': 0.0,
                            'confidence': 0.0,
                            'reason': 'No analysis available'
                        }
                
                return validated_scores
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: {'score': 0.0, 'confidence': 0.0, 'reason': 'No JSON response'} for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse structured scores from response: {e}")
            return {etf: {'score': 0.0, 'confidence': 0.0, 'reason': f'Parse error: {str(e)}'} for etf in universe}


# Example usage and testing
if __name__ == "__main__":
    print("Macro Economist Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = MacroEconomistAgent("TestMacroEconomist")
        print(f"✓ Macro economist agent initialized")
        print(f"  Specialization: {agent.specialization}")
        print(f"  Provider: {agent.get_provider_info()['provider']}")
        
        # Test with sample state
        sample_state = {
            'macro_data': {
                'CPIAUCSL': {'latest_value': 300.0, 'periods': 12},
                'UNRATE': {'latest_value': 3.5, 'periods': 12}
            },
            'etf_data': pd.DataFrame({
                'SPY': [100, 101, 102, 103, 104],
                'QQQ': [200, 201, 202, 203, 204],
                'TLT': [150, 149, 148, 147, 146]
            }),
            'universe': ['SPY', 'QQQ', 'TLT'],
            'analyst_scores': {}
        }
        
        # Test analysis
        result_state = agent.analyze(sample_state)
        print(f"✓ Analysis completed")
        print(f"  Macro scores: {result_state.get('macro_scores', {})}")
        
        # Show detailed output for first ETF
        macro_scores = result_state.get('macro_scores', {})
        if macro_scores:
            first_etf = list(macro_scores.keys())[0]
            etf_data = macro_scores[first_etf]
            print(f"  Sample ETF ({first_etf}):")
            print(f"    Score: {etf_data.get('score', 0.0)}")
            print(f"    Confidence: {etf_data.get('confidence', 0.0)}")
            print(f"    Reason: {etf_data.get('reason', 'No reason')[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Macro economist test completed!")

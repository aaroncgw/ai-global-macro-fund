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
        Analyze macroeconomic data and score ETFs.
        
        Args:
            state: LangGraph state dictionary containing:
                - macro_data: Macro economic indicators
                - etf_data: ETF price data
                - universe: List of ETFs to analyze
                - analyst_scores: Dictionary to store scores
                
        Returns:
            Updated state dictionary with macro economist scores
        """
        try:
            # Extract data from state
            macro_data = state.get('macro_data', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            # Ensure analyst_scores exists in state
            if 'analyst_scores' not in state:
                state['analyst_scores'] = {}
            
            # Create analysis prompt
            prompt = self._create_macro_analysis_prompt(macro_data, etf_data, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse scores from response
            scores = self._parse_scores(response, universe)
            
            # Store scores in state
            state['analyst_scores']['macro'] = scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['macro_economist'] = {
                'scores': scores,
                'reasoning': f"Macro economist analysis based on {len(macro_data)} indicators and {len(universe)} ETFs",
                'key_factors': list(macro_data.keys()) if macro_data else ['No macro data available'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Macro economist analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Macro economist analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: 0.0 for etf in state.get('universe', [])}
            state['analyst_scores']['macro'] = neutral_scores
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
        # Calculate ETF returns for analysis
        etf_returns = etf_data.pct_change().tail(252) if etf_data is not None and not etf_data.empty else pd.DataFrame()
        
        # Format macro indicators
        macro_summary = self._format_macro_indicators(macro_data)
        
        # Format ETF returns
        etf_summary = self._format_etf_returns(etf_returns, universe)
        
        prompt = f"""
        As a macro economist, analyze the following data and score each ETF from -1 (strong sell) to 1 (strong buy):
        
        MACRO ECONOMIC DATA:
        {macro_summary}
        
        ETF RETURNS (252-day lookback):
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
                    recent_return = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
                    avg_return = float(returns.mean())
                    formatted.append(f"- {etf}: Recent: {recent_return:.3f}, Avg: {avg_return:.3f}")
                else:
                    formatted.append(f"- {etf}: No returns data")
            else:
                formatted.append(f"- {etf}: Not in data")
        
        return '\n'.join(formatted) if formatted else "No ETF returns data available"
    
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
        print(f"  Scores: {result_state.get('analyst_scores', {}).get('macro', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Macro economist test completed!")

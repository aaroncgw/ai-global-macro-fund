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
        Now processes ETFs one by one for more consistent scoring.
        
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
            
            # Process each ETF individually for more consistent scoring
            macro_scores = {}
            
            for etf in universe:
                try:
                    logger.info(f"Processing {etf} for macro economist analysis...")
                    
                    # Create individual ETF analysis prompt
                    prompt = self._create_individual_etf_macro_prompt(macro_data, etf_data, etf)
                    
                    # Get LLM response with JSON format
                    response = self.llm(prompt, response_format='json_object')
                    
                    # Parse the structured response for this single ETF
                    etf_score = self._parse_single_etf_score(response, etf)
                    macro_scores[etf] = etf_score
                    
                    logger.debug(f"Completed macro analysis for {etf}: score={etf_score.get('score', 0.0)}")
                    
                except Exception as etf_error:
                    logger.error(f"Failed to analyze {etf}: {etf_error}")
                    # Provide neutral score for failed ETF
                    macro_scores[etf] = {
                        'score': 0.0,
                        'confidence': 0.0,
                        'reason': f'Analysis failed for {etf}: {str(etf_error)}'
                    }
            
            # Store macro scores in state
            state['macro_scores'] = macro_scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['macro_economist'] = {
                'macro_scores': macro_scores,
                'reasoning': f"Macro economist analysis completed for {len(universe)} ETFs (processed individually)",
                'key_factors': list(macro_data.keys()) if macro_data else ['No macro data available'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Macro economist analysis completed for {len(universe)} ETFs (individual processing)")
            return state
            
        except Exception as e:
            logger.error(f"Macro economist analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: {'score': 0.0, 'confidence': 0.0, 'reason': 'Analysis failed due to error'} for etf in state.get('universe', [])}
            state['macro_scores'] = neutral_scores
            return state
    
    
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
    
    
    
    
    def _create_individual_etf_macro_prompt(self, macro_data: dict, etf_data: pd.DataFrame, etf: str) -> str:
        """
        Create a prompt for individual ETF macroeconomic analysis.
        
        Args:
            macro_data: Dictionary of macro economic indicators
            etf_data: DataFrame with ETF price data
            etf: Single ETF symbol to analyze
            
        Returns:
            Formatted prompt string for single ETF analysis
        """
        # Calculate ETF returns for this specific ETF
        etf_returns = None
        if etf_data is not None and not etf_data.empty and etf in etf_data.columns:
            etf_returns = etf_data[etf].pct_change().dropna()
        
        # Format macro indicators
        macro_summary = self._format_macro_indicators(macro_data)
        
        # Format ETF-specific data
        etf_summary = self._format_individual_etf_data(etf_returns, etf)
        
        prompt = f"""
        As a macro economist, analyze the following data and score ONLY the ETF {etf} from -1 (strong sell) to 1 (strong buy).
        
        MACRO ECONOMIC DATA:
        {macro_summary}
        
        {etf} SPECIFIC DATA:
        {etf_summary}
        
        ANALYSIS FRAMEWORK & REQUIREMENTS FOR {etf}:
        1. Inflation Impact: How does current inflation affect {etf} specifically?
        2. Interest Rate Environment: Impact on {etf} based on its asset class
        3. Economic Growth: How does current growth affect {etf}?
        4. Currency Strength: Impact on {etf} if it's international/currency-focused
        5. Economic Cycle Position: Where are we in the cycle and how does it affect {etf}?
        
        For {etf}, provide:
        - Score from -1.0 (strong sell) to 1.0 (strong buy)
        - Confidence level from 0.0 (no confidence) to 1.0 (high confidence)
        - Detailed reasoning based on macro factors above
        - Focus on {DEFAULT_HORIZON} horizon
        
        CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
        {{"{etf}": {{"score": -1 to 1, "confidence": 0-1, "reason": "detailed explanation"}}}}
        
        Do not include any text before or after the JSON. Return only the JSON object.
        """
        return prompt
    
    def _format_individual_etf_data(self, etf_returns: pd.Series, etf: str) -> str:
        """Format individual ETF data for the prompt."""
        if etf_returns is None or etf_returns.empty:
            return f"No {etf} returns data available"
        
        try:
            # Safely extract scalar values
            recent_return = etf_returns.iloc[-1]
            if hasattr(recent_return, 'item'):
                recent_return = recent_return.item()
            recent_return = float(recent_return)
        except (ValueError, TypeError, IndexError):
            recent_return = 0.0
        
        try:
            avg_return = etf_returns.mean()
            if hasattr(avg_return, 'item'):
                avg_return = avg_return.item()
            avg_return = float(avg_return)
        except (ValueError, TypeError):
            avg_return = 0.0
        
        try:
            volatility = etf_returns.std() * (252 ** 0.5)  # Annualized volatility
            if hasattr(volatility, 'item'):
                volatility = volatility.item()
            volatility = float(volatility)
        except (ValueError, TypeError):
            volatility = 0.0
        
        return f"- {etf}: Recent Return: {recent_return:.3f}, Avg Return: {avg_return:.3f}, Volatility: {volatility:.3f}"
    
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

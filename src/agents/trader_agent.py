"""
Trader Agent for ETF Allocation Proposals

This agent proposes initial buy/sell allocations for ETFs based on
debate results and analyst scores from the macro analysis pipeline.
"""

from src.agents.base_agent import BaseAgent
import json
import logging

logger = logging.getLogger(__name__)


class TraderAgent(BaseAgent):
    """
    Trader Agent that proposes initial ETF allocations based on analysis.
    
    This agent focuses on:
    - Converting analysis into actionable allocations
    - Proposing buy/sell decisions for ETFs
    - Considering debate results and analyst scores
    - Creating initial portfolio positioning
    - Balancing risk and return objectives
    """
    
    def __init__(self, agent_name: str = "TraderAgent"):
        """Initialize the trader agent."""
        super().__init__(agent_name)
        self.specialization = "allocation_proposal"
        self.analysis_focus = "trading_decisions"
        self.role = "trader"
    
    def analyze(self, data: dict) -> dict:
        """
        Analyze data and propose ETF allocations.
        
        Args:
            data: Dictionary containing debate results, analyst scores, and universe
            
        Returns:
            Analysis results with proposed allocations
        """
        try:
            debate_output = data.get('debate_output', [])
            analyst_scores = data.get('analyst_scores', {})
            universe = data.get('universe', [])
            
            # Create allocation proposal prompt
            prompt = self._create_allocation_prompt(debate_output, analyst_scores, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse allocations from response
            allocations = self._parse_allocations(response, universe)
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'proposed_allocations': allocations,
                'reasoning': f"Trader proposal based on {len(debate_output)} debate rounds and {len(analyst_scores)} analyst scores",
                'key_factors': list(analyst_scores.keys()) if analyst_scores else ['No analyst scores available'],
                'llm_response': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Trader analysis failed: {e}")
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'error': str(e),
                'status': 'failed'
            }
    
    def propose(self, state: dict) -> dict:
        """
        Propose initial ETF allocations based on analysis.
        
        Args:
            state: LangGraph state dictionary containing:
                - debate_output: Results from debate researchers
                - analyst_scores: Scores from macro analyst agents
                - universe: List of ETFs to allocate
                
        Returns:
            Updated state dictionary with proposed allocations
        """
        try:
            # Extract data from state
            debate_output = state.get('debate_output', [])
            analyst_scores = state.get('analyst_scores', {})
            universe = state.get('universe', [])
            
            # Create allocation proposal prompt
            prompt = self._create_allocation_prompt(debate_output, analyst_scores, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse allocations from response
            allocations = self._parse_allocations(response, universe)
            
            # Store proposed allocations in state
            state['proposed_allocations'] = allocations
            
            logger.info(f"Trader proposed allocations for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Trader proposal failed: {e}")
            # Return neutral allocations on error
            neutral_allocations = {etf: 1.0 / len(state.get('universe', [])) for etf in state.get('universe', [])}
            state['proposed_allocations'] = neutral_allocations
            return state
    
    def _create_allocation_prompt(self, debate_output: list, analyst_scores: dict, universe: list) -> str:
        """
        Create a prompt for ETF allocation proposals.
        
        Args:
            debate_output: List of debate results
            analyst_scores: Dictionary of analyst scores
            universe: List of ETFs to allocate
            
        Returns:
            Formatted prompt string
        """
        # Format debate results
        debate_summary = self._format_debate_results(debate_output)
        
        # Format analyst scores
        scores_summary = self._format_analyst_scores(analyst_scores)
        
        prompt = f"""
        As a professional trader, propose initial buy/sell allocations for ETFs based on the following analysis:
        
        DEBATE RESULTS:
        {debate_summary}
        
        ANALYST SCORES:
        {scores_summary}
        
        ETF UNIVERSE:
        {', '.join(universe)}
        
        ALLOCATION REQUIREMENTS:
        1. Convert analysis into actionable allocations
        2. Propose buy/sell decisions for each ETF
        3. Consider both bullish and bearish arguments from debates
        4. Balance risk and return objectives
        5. Ensure allocations sum to 100% (or close to it)
        6. Focus on macro trends and economic indicators
        7. Consider geopolitical risks and opportunities
        8. Account for correlation and diversification benefits
        
        ALLOCATION GUIDELINES:
        - Positive scores suggest higher allocations
        - Negative scores suggest lower allocations or short positions
        - Consider debate consensus and disagreements
        - Balance growth opportunities with risk management
        - Account for macro environment and market conditions
        
        Return ONLY a JSON object with ETF allocations as percentages:
        {{"SPY": 25.0, "QQQ": 20.0, "TLT": 15.0, "GLD": 10.0, ...}}
        """
        return prompt
    
    def _format_debate_results(self, debate_output: list) -> str:
        """Format debate results for the prompt."""
        if not debate_output:
            return "No debate results available"
        
        # Take first 10 lines to avoid overly long prompts
        summary_lines = debate_output[:10]
        return '\n'.join(summary_lines)
    
    def _format_analyst_scores(self, analyst_scores: dict) -> str:
        """Format analyst scores for the prompt."""
        if not analyst_scores:
            return "No analyst scores available"
        
        formatted = []
        for analyst, scores in analyst_scores.items():
            formatted.append(f"\n{analyst.upper()} SCORES:")
            for etf, score in scores.items():
                formatted.append(f"  {etf}: {score:.3f}")
        
        return '\n'.join(formatted)
    
    def _parse_allocations(self, response: str, universe: list) -> dict:
        """
        Parse ETF allocations from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to allocate
            
        Returns:
            Dictionary with ETF allocations
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                allocations = json.loads(json_str)
                
                # Validate and normalize allocations
                validated_allocations = {}
                total_allocation = 0.0
                
                for etf in universe:
                    if etf in allocations:
                        allocation = float(allocations[etf])
                        # Ensure non-negative allocations
                        allocation = max(0.0, allocation)
                        validated_allocations[etf] = allocation
                        total_allocation += allocation
                    else:
                        validated_allocations[etf] = 0.0
                
                # Normalize allocations to sum to 100%
                if total_allocation > 0:
                    for etf in validated_allocations:
                        validated_allocations[etf] = (validated_allocations[etf] / total_allocation) * 100.0
                else:
                    # Equal allocation if no valid allocations
                    equal_allocation = 100.0 / len(universe)
                    for etf in universe:
                        validated_allocations[etf] = equal_allocation
                
                return validated_allocations
            else:
                logger.warning("No JSON found in LLM response")
                # Return equal allocations
                equal_allocation = 100.0 / len(universe)
                return {etf: equal_allocation for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse allocations from response: {e}")
            # Return equal allocations on error
            equal_allocation = 100.0 / len(universe)
            return {etf: equal_allocation for etf in universe}


# Example usage and testing
if __name__ == "__main__":
    print("Trader Agent Test")
    print("="*30)
    
    try:
        # Initialize trader agent
        trader = TraderAgent("TestTrader")
        print(f"✓ Trader agent initialized")
        print(f"  Specialization: {trader.specialization}")
        print(f"  Provider: {trader.get_provider_info()['provider']}")
        
        # Test with sample state
        sample_state = {
            'debate_output': [
                '=== MACRO ETF DEBATE STARTED ===',
                'Universe: SPY, QQQ, TLT, GLD',
                'Analysts: macro, geo, correlation',
                '=== ROUND 1 ===',
                'BULLISH ARGUMENT: Strong macro trends support growth assets...',
                'BEARISH COUNTER-ARGUMENT: Geopolitical risks create headwinds...'
            ],
            'analyst_scores': {
                'macro': {'SPY': 0.4, 'QQQ': 0.6, 'TLT': -0.8, 'GLD': 0.7},
                'geo': {'SPY': -0.4, 'QQQ': -0.6, 'TLT': 0.3, 'GLD': 0.7},
                'correlation': {'SPY': 0.1, 'QQQ': 0.0, 'TLT': 0.6, 'GLD': 0.7}
            },
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Test allocation proposal
        result_state = trader.propose(sample_state)
        print(f"✓ Allocation proposal completed")
        print(f"  Proposed allocations: {result_state.get('proposed_allocations', {})}")
        
        # Test individual analysis
        analysis_result = trader.analyze(sample_state)
        print(f"✓ Individual analysis completed")
        print(f"  Allocations: {analysis_result.get('proposed_allocations', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*30)
    print("Trader agent test completed!")

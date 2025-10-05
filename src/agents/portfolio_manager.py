"""
Portfolio Manager Agent for ETF Final Allocation Decisions

This agent makes final trading decisions and generates allocations using LLM-driven
synthesis like the original ai-hedge-fund PortfolioManager, but for batch ETFs.
"""

import json
import logging
import pandas as pd
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent that makes final trading decisions using LLM synthesis.
    
    This agent focuses on:
    - LLM-driven portfolio synthesis and allocation decisions
    - Aggregating analyst scores and risk metrics
    - Buy/sell/hold recommendations with reasoning
    - Portfolio construction and position sizing
    """
    
    def __init__(self, agent_name: str = "PortfolioManagerAgent"):
        """Initialize the portfolio manager agent."""
        super().__init__(agent_name)
        self.specialization = "portfolio_management"
        self.analysis_focus = "final_allocations_and_recommendations"
        self.role = "portfolio_manager"
    
    def analyze(self, state: dict) -> dict:
        """
        Analyze and manage portfolio (required by BaseAgent).
        
        Args:
            state: LangGraph state dictionary
            
        Returns:
            Updated state dictionary with final allocations
        """
        return self.manage(state)
    
    def manage(self, state: dict) -> dict:
        """
        Manage portfolio using LLM-driven synthesis of analyst scores and risk metrics.
        
        Args:
            state: LangGraph state dictionary containing:
                - scores: Aggregated analyst scores from all agents
                - risk_metrics: Risk assessments from risk manager
                - universe: List of ETFs to manage
                
        Returns:
            Updated state dictionary with final allocations
        """
        try:
            # Extract data from state - get individual analyst scores
            macro_scores = state.get('macro_scores', {})
            geo_scores = state.get('geo_scores', {})
            risks = state.get('risk_metrics', {})
            universe = state.get('universe', [])
            
            if not macro_scores or not geo_scores or not risks or not universe:
                logger.warning("Insufficient data for portfolio management")
                neutral_allocations = {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'Insufficient data'} for etf in universe}
                state['final_allocations'] = neutral_allocations
                return state
            
            # Get detailed reasoning from all agents
            agent_reasoning = state.get('agent_reasoning', {})
            
            # Calculate position limit (max 20% of universe)
            max_positions = max(1, int(len(universe) * 0.2))
            
            # Create LLM prompt for portfolio synthesis with full reasoning
            prompt = f"""
            As a hedge fund manager, synthesize the following individual analyst scores, detailed reasoning, and risk metrics to recommend portfolio allocations for the ETF universe {universe}.
            
            MACRO ECONOMIST SCORES:
            {json.dumps(macro_scores, indent=2)}
            
            GEOPOLITICAL ANALYST SCORES:
            {json.dumps(geo_scores, indent=2)}
            
            DETAILED ANALYST REASONING:
            {self._format_agent_reasoning(agent_reasoning)}
            
            RISK METRICS:
            {json.dumps(risks, indent=2)}
            
            PORTFOLIO SYNTHESIS REQUIREMENTS:
            1. AGGREGATE ANALYST SCORES: Combine macro economist and geopolitical analyst scores for each ETF
            2. WEIGHT ANALYSTS: Decide how much weight to give each analyst based on their confidence and reasoning quality
            3. Consider both individual analyst scores and risk levels for each ETF
            4. Use the detailed reasoning from each analyst to understand their logic
            5. Balance return potential with risk management
            6. Ensure total allocations sum to 1.0 (100% of portfolio)
            7. Consider correlations implicitly through reasoning
            8. Provide clear reasoning for each allocation decision that references the analyst insights
            9. POSITION LIMIT: Allocate to maximum {max_positions} ETFs only (20% of universe size)
            10. Focus on highest conviction opportunities with best risk-adjusted returns
            
            ALLOCATION GUIDELINES:
            - High scores + Low risk = Higher allocation
            - High scores + High risk = Moderate allocation with risk management
            - Low scores + Any risk = Lower allocation or hold
            - Negative scores = Consider sell or minimal allocation
            
            RISK CONSIDERATIONS:
            - Low risk: Can support higher allocations
            - Medium risk: Moderate allocations with monitoring
            - High risk: Lower allocations or avoid
            
            COMPREHENSIVE REASONING REQUIREMENTS:
            For each allocation decision, provide detailed reasoning that includes:
            1. Analyst consensus analysis (what the macro and geopolitical analysts concluded)
            2. Risk assessment integration (how risk manager's assessment influenced the decision)
            3. Score aggregation logic (how individual analyst scores were weighted and combined)
            4. Risk-return trade-off analysis (balancing potential returns with identified risks)
            5. Portfolio construction rationale (why this allocation fits the overall portfolio)
            6. Conviction level explanation (why this position size was chosen)
            7. Risk management considerations (how risks are being managed)
            8. Market outlook integration (how current market conditions influenced the decision)
            
            EXAMPLES OF EXPECTED OUTPUT:
            Example 1 - Risk-off environment:
            {{"SPY": {{"action": "sell", "allocation": 0.0, "reason": "Risk-off sentiment with high volatility and geopolitical tensions favor defensive positioning"}}, "TLT": {{"action": "buy", "allocation": 1.0, "reason": "Flight-to-quality flows and safe-haven demand support treasury allocation"}}}}
            
            Example 2 - Growth environment:
            {{"SPY": {{"action": "buy", "allocation": 0.7, "reason": "Strong economic growth and accommodative policy support equity allocation"}}, "QQQ": {{"action": "buy", "allocation": 0.3, "reason": "Technology sector benefits from growth environment and innovation trends"}}}}
            
            Example 3 - Balanced approach:
            {{"SPY": {{"action": "buy", "allocation": 0.4, "reason": "Moderate equity exposure with diversification benefits"}}, "TLT": {{"action": "buy", "allocation": 0.3, "reason": "Bond allocation for income and portfolio stability"}}, "GLD": {{"action": "buy", "allocation": 0.3, "reason": "Gold allocation for inflation hedge and diversification"}}}}
            
            CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
            {{"ETF": {{"action": "buy/sell/hold", "allocation": 0.0-1.0, "reason": "comprehensive detailed explanation covering all reasoning requirements above"}}}}
            
            Do not include any text before or after the JSON. Return only the JSON object.
            Ensure all ETF allocations sum to 1.0, focus on top {max_positions} opportunities, and provide comprehensive reasoning for each decision.
            """
            
            # Get LLM response with JSON format
            response = self.llm(prompt, response_format='json_object')
            
            # Parse the structured response
            final_allocations = self._parse_allocations(response, universe)
            
            # Enforce position limit (max 20% of universe)
            final_allocations = self._enforce_position_limit(final_allocations, max_positions)
            
            # Store final allocations in state
            state['final_allocations'] = final_allocations
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['portfolio_manager'] = {
                'final_allocations': final_allocations,
                'reasoning': f"LLM-driven portfolio synthesis aggregating macro and geopolitical analyst scores and risk metrics for {len(universe)} ETFs",
                'synthesis_method': 'LLM synthesis with score aggregation and risk consideration',
                'performance_metrics': {
                    'total_allocation': sum([alloc.get('allocation', 0) for alloc in final_allocations.values()]),
                    'number_of_positions': len([etf for etf, alloc in final_allocations.items() if alloc.get('allocation', 0) > 0]),
                    'buy_positions': len([etf for etf, alloc in final_allocations.items() if alloc.get('action') == 'buy']),
                    'hold_positions': len([etf for etf, alloc in final_allocations.items() if alloc.get('action') == 'hold']),
                    'sell_positions': len([etf for etf, alloc in final_allocations.items() if alloc.get('action') == 'sell'])
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"LLM-driven portfolio management completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Portfolio management failed: {e}")
            # Return neutral allocations on error
            neutral_allocations = {etf: {'action': 'hold', 'allocation': 0.0, 'reason': f'Portfolio management failed: {str(e)}'} for etf in state.get('universe', [])}
            state['final_allocations'] = neutral_allocations
            return state
    
    def _parse_allocations(self, response: str, universe: list) -> dict:
        """
        Parse structured allocations from LLM response.
        
        Args:
            response: LLM response string
            universe: List of ETFs to allocate
            
        Returns:
            Dictionary with final allocations
        """
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                # Find JSON part
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                allocations = json.loads(json_str)
                
                # Validate and structure allocations
                validated_allocations = {}
                for etf in universe:
                    if etf in allocations and isinstance(allocations[etf], dict):
                        etf_data = allocations[etf]
                        action = str(etf_data.get('action', 'hold')).lower()
                        allocation = float(etf_data.get('allocation', 0.0))
                        reason = str(etf_data.get('reason', 'No reasoning provided'))
                        
                        # Validate action
                        if action not in ['buy', 'sell', 'hold']:
                            action = 'hold'
                        
                        # Clamp allocation to [0, 1]
                        allocation = max(0.0, min(1.0, allocation))
                        
                        validated_allocations[etf] = {
                            'action': action,
                            'allocation': allocation,
                            'reason': reason
                        }
                    else:
                        # Default values for missing ETFs
                        validated_allocations[etf] = {
                            'action': 'hold',
                            'allocation': 0.0,
                            'reason': 'No allocation provided'
                        }
                
                # Normalize allocations to sum to 1.0
                total_allocation = sum([alloc['allocation'] for alloc in validated_allocations.values()])
                if total_allocation > 0:
                    for etf in validated_allocations:
                        validated_allocations[etf]['allocation'] /= total_allocation
                
                return validated_allocations
            else:
                logger.warning("No JSON found in LLM response")
                return {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'No JSON response'} for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse allocations from response: {e}")
            return {etf: {'action': 'hold', 'allocation': 0.0, 'reason': f'Parse error: {str(e)}'} for etf in universe}
    
    def _format_agent_reasoning(self, agent_reasoning: dict) -> str:
        """
        Format detailed reasoning from all agents for the LLM prompt.
        
        Args:
            agent_reasoning: Dictionary with reasoning from all agents
            
        Returns:
            Formatted string with all agent reasoning
        """
        if not agent_reasoning:
            return "No detailed reasoning available from agents"
        
        formatted = []
        for agent_name, reasoning_data in agent_reasoning.items():
            if isinstance(reasoning_data, dict):
                formatted.append(f"\n{agent_name.upper().replace('_', ' ')} REASONING:")
                formatted.append(f"  Summary: {reasoning_data.get('reasoning', 'No summary available')}")
                
                # Add specific scores/reasoning for each ETF
                if 'macro_scores' in reasoning_data:
                    scores = reasoning_data['macro_scores']
                    formatted.append("  ETF Analysis:")
                    for etf, data in scores.items():
                        if isinstance(data, dict):
                            score = data.get('score', 0)
                            confidence = data.get('confidence', 0)
                            reason = data.get('reason', 'No reasoning')
                            formatted.append(f"    {etf}: Score {score:.3f} (Confidence: {confidence:.1%}) - {reason}")
                
                elif 'geo_scores' in reasoning_data:
                    scores = reasoning_data['geo_scores']
                    formatted.append("  ETF Analysis:")
                    for etf, data in scores.items():
                        if isinstance(data, dict):
                            score = data.get('score', 0)
                            confidence = data.get('confidence', 0)
                            reason = data.get('reason', 'No reasoning')
                            formatted.append(f"    {etf}: Score {score:.3f} (Confidence: {confidence:.1%}) - {reason}")
                
                elif 'risk_metrics' in reasoning_data:
                    metrics = reasoning_data['risk_metrics']
                    formatted.append("  Risk Assessment:")
                    for etf, data in metrics.items():
                        if isinstance(data, dict):
                            risk_level = data.get('risk_level', 'unknown')
                            volatility = data.get('volatility', 0)
                            reason = data.get('reason', 'No reasoning')
                            formatted.append(f"    {etf}: {risk_level.upper()} Risk (Vol: {volatility:.1%}) - {reason}")
                
                # Add key factors
                key_factors = reasoning_data.get('key_factors', [])
                if key_factors:
                    formatted.append(f"  Key Factors: {', '.join(key_factors[:5])}")
        
        return '\n'.join(formatted) if formatted else "No detailed reasoning available"
    
    def _enforce_position_limit(self, allocations: dict, max_positions: int) -> dict:
        """
        Enforce position limit by keeping only the top allocations.
        
        Args:
            allocations: Dictionary with ETF allocations
            max_positions: Maximum number of positions allowed
            
        Returns:
            Dictionary with position limit enforced
        """
        try:
            # Filter out zero allocations and sort by allocation size
            active_allocations = {etf: data for etf, data in allocations.items() 
                                if data.get('allocation', 0) > 0}
            
            if len(active_allocations) <= max_positions:
                return allocations
            
            # Sort by allocation size (descending)
            sorted_allocations = sorted(active_allocations.items(), 
                                     key=lambda x: x[1].get('allocation', 0), 
                                     reverse=True)
            
            # Keep only top positions
            top_allocations = dict(sorted_allocations[:max_positions])
            
            # Set all other positions to hold with 0 allocation
            result = {}
            for etf, data in allocations.items():
                if etf in top_allocations:
                    result[etf] = data
                else:
                    result[etf] = {
                        'action': 'hold',
                        'allocation': 0.0,
                        'reason': f'Position limit enforced - not in top {max_positions} opportunities'
                    }
            
            # Renormalize allocations to sum to 1.0
            total_allocation = sum(data.get('allocation', 0) for data in result.values())
            if total_allocation > 0:
                for etf in result:
                    if result[etf].get('allocation', 0) > 0:
                        result[etf]['allocation'] /= total_allocation
            
            logger.info(f"Position limit enforced: {len(top_allocations)} positions from {len(active_allocations)} candidates")
            return result
            
        except Exception as e:
            logger.error(f"Position limit enforcement failed: {e}")
            return allocations


# Example usage and testing
if __name__ == "__main__":
    print("Portfolio Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = PortfolioManagerAgent("TestPortfolioManagerAgent")
        print(f"✓ Portfolio manager agent initialized")
        print(f"  Specialization: {agent.specialization}")
        print(f"  Provider: {agent.get_provider_info()['provider']}")
        
        # Test with sample state
        import pandas as pd
        import numpy as np
        
        # Create sample ETF data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        etf_data = pd.DataFrame({
            'SPY': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'TLT': 150 + np.cumsum(np.random.randn(100) * 0.005),
            'GLD': 180 + np.cumsum(np.random.randn(100) * 0.008)
        }, index=dates)
        
        sample_state = {
            'scores': {
                'macro_economist': {
                    'SPY': {'score': 0.3, 'confidence': 0.8, 'reason': 'Positive macro trends'},
                    'TLT': {'score': -0.2, 'confidence': 0.6, 'reason': 'Rising rates pressure'},
                    'GLD': {'score': 0.1, 'confidence': 0.5, 'reason': 'Inflation hedge potential'}
                },
                'geopolitical_analyst': {
                    'SPY': {'score': 0.1, 'confidence': 0.5, 'reason': 'US stability'},
                    'TLT': {'score': 0.4, 'confidence': 0.7, 'reason': 'Safe haven demand'},
                    'GLD': {'score': 0.2, 'confidence': 0.6, 'reason': 'Geopolitical tensions'}
                }
            },
            'risk_metrics': {
                'SPY': {
                    'risk_level': 'low',
                    'volatility': 0.15,
                    'reason': 'Stable large-cap exposure with low volatility'
                },
                'TLT': {
                    'risk_level': 'high',
                    'volatility': 0.35,
                    'reason': 'Interest rate sensitivity creates volatility risk'
                },
                'GLD': {
                    'risk_level': 'medium',
                    'volatility': 0.20,
                    'reason': 'Safe haven demand but commodity volatility'
                }
            },
            'etf_data': etf_data,
            'universe': ['SPY', 'TLT', 'GLD']
        }
        
        # Test portfolio management
        result_state = agent.manage(sample_state)
        print(f"✓ LLM-driven portfolio management completed")
        print(f"  Final allocations: {result_state.get('final_allocations', {})}")
        
        # Show detailed output for each ETF
        final_allocations = result_state.get('final_allocations', {})
        print("\nPortfolio Summary:")
        print("="*20)
        for etf, allocation in final_allocations.items():
            if isinstance(allocation, dict):
                action = allocation.get('action', 'unknown')
                allocation_pct = allocation.get('allocation', 0.0)
                reason = allocation.get('reason', 'No reasoning provided')
                print(f"{etf}: {action.upper()} {allocation_pct:.1%} - {reason[:50]}...")
            else:
                print(f"{etf}: {allocation}")
        
        print("\n✓ Portfolio manager agent test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

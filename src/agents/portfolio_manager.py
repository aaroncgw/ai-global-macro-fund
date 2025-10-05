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
            # Extract data from state
            scores = state.get('scores', {})
            risks = state.get('risk_metrics', {})
            universe = state.get('universe', [])
            
            if not scores or not risks or not universe:
                logger.warning("Insufficient data for portfolio management")
                neutral_allocations = {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'Insufficient data'} for etf in universe}
                state['final_allocations'] = neutral_allocations
                return state
            
            # Aggregate scores generically (average per ETF from all agents)
            avg_scores = {}
            for etf in universe:
                etf_scores = []
                for agent_name, agent_scores in scores.items():
                    if isinstance(agent_scores, dict) and etf in agent_scores:
                        etf_score = agent_scores[etf].get('score', 0) if isinstance(agent_scores[etf], dict) else 0
                        etf_scores.append(etf_score)
                
                avg_score = sum(etf_scores) / len(etf_scores) if etf_scores else 0
                avg_scores[etf] = avg_score
            
            # Create LLM prompt for portfolio synthesis
            prompt = f"""
            As a hedge fund manager, synthesize the following analyst scores and risk metrics to recommend portfolio allocations for the ETF universe {universe}.
            
            AVERAGED ANALYST SCORES (from all agents):
            {json.dumps(avg_scores, indent=2)}
            
            RISK METRICS:
            {json.dumps(risks, indent=2)}
            
            PORTFOLIO SYNTHESIS REQUIREMENTS:
            1. Consider both analyst scores and risk levels for each ETF
            2. Balance return potential with risk management
            3. Ensure total allocations sum to 1.0 (100% of portfolio)
            4. Consider correlations implicitly through reasoning
            5. Provide clear reasoning for each allocation decision
            
            ALLOCATION GUIDELINES:
            - High scores + Low risk = Higher allocation
            - High scores + High risk = Moderate allocation with risk management
            - Low scores + Any risk = Lower allocation or hold
            - Negative scores = Consider sell or minimal allocation
            
            RISK CONSIDERATIONS:
            - Low risk: Can support higher allocations
            - Medium risk: Moderate allocations with monitoring
            - High risk: Lower allocations or avoid
            
            Output format (JSON):
            {{"ETF": {{"action": "buy/sell/hold", "allocation": 0.0-1.0, "reason": "detailed explanation"}}}}
            
            Ensure all ETF allocations sum to 1.0 and provide clear reasoning for each decision.
            """
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse the structured response
            final_allocations = self._parse_allocations(response, universe)
            
            # Store final allocations in state
            state['final_allocations'] = final_allocations
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['portfolio_agent'] = {
                'final_allocations': final_allocations,
                'reasoning': f"LLM-driven portfolio synthesis aggregating {len(scores)} analyst scores and risk metrics for {len(universe)} ETFs",
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
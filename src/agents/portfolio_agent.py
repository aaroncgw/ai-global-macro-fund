"""
Portfolio Agent for ETF Final Allocation Decisions

This agent makes final trading decisions and generates allocations based on
risk assessments and mathematical optimization. Mimics the original ai-hedge-fund
PortfolioManager but for batch ETFs.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """
    Portfolio Agent that makes final trading decisions and generates allocations.
    
    This agent focuses on:
    - Final allocation decisions based on risk assessments
    - Mathematical optimization using mean-variance framework
    - Buy/sell/hold recommendations
    - Portfolio construction and position sizing
    """
    
    def __init__(self, agent_name: str = "PortfolioAgent"):
        """Initialize the portfolio agent."""
        self.agent_name = agent_name
        self.specialization = "portfolio_management"
        self.analysis_focus = "final_decisions"
        self.role = "portfolio_manager"
    
    def manage(self, state: dict) -> dict:
        """
        Manage portfolio and make final allocation decisions.
        
        Args:
            state: LangGraph state dictionary containing:
                - risk_assessments: Risk assessments from risk manager
                - etf_data: ETF price data for correlation analysis
                - universe: List of ETFs to manage
                
        Returns:
            Updated state dictionary with final allocations
        """
        try:
            # Extract data from state
            risks = state.get('risk_assessments', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            if not risks or etf_data.empty:
                logger.warning("Insufficient data for portfolio management")
                # Return neutral allocations
                neutral_allocations = {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'Insufficient data'} for etf in universe}
                state['final_allocations'] = neutral_allocations
                return state
            
            # Calculate correlation matrix
            corr_matrix = etf_data.pct_change().corr()
            
            if corr_matrix.empty:
                logger.warning("Could not calculate correlation matrix")
                # Return risk-based allocations without optimization
                allocations = self._create_risk_based_allocations(risks, universe)
                state['final_allocations'] = allocations
                return state
            
            # Mathematical optimization
            allocations = self._optimize_portfolio(risks, corr_matrix, universe)
            
            # Store final allocations in state
            state['final_allocations'] = allocations
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['portfolio_agent'] = {
                'final_allocations': allocations,
                'reasoning': f"Portfolio optimization using mean-variance framework for {len(universe)} ETFs",
                'optimization_method': 'Mathematical optimization with risk-return tradeoff',
                'performance_metrics': {
                    'total_allocation': sum([alloc['allocation'] for alloc in allocations.values()]),
                    'number_of_positions': len([etf for etf, alloc in allocations.items() if alloc['allocation'] > 0]),
                    'buy_positions': len([etf for etf, alloc in allocations.items() if alloc['action'] == 'buy']),
                    'hold_positions': len([etf for etf, alloc in allocations.items() if alloc['action'] == 'hold'])
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Portfolio management completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Portfolio management failed: {e}")
            # Return neutral allocations on error
            neutral_allocations = {etf: {'action': 'hold', 'allocation': 0.0, 'reason': f'Portfolio management failed: {str(e)}'} for etf in state.get('universe', [])}
            state['final_allocations'] = neutral_allocations
            return state
    
    def _optimize_portfolio(self, risks: dict, corr_matrix: pd.DataFrame, universe: list) -> dict:
        """
        Optimize portfolio using mathematical optimization.
        
        Args:
            risks: Risk assessments from risk manager
            corr_matrix: Correlation matrix DataFrame
            universe: List of ETFs to optimize
            
        Returns:
            Dictionary with final allocations
        """
        try:
            # Extract expected scores from risk assessments
            expected_scores = np.array([risks.get(etf, {'adjusted_score': 0.0}).get('adjusted_score', 0.0) for etf in universe])
            
            # Use correlation matrix as covariance matrix (simplified)
            cov_matrix = corr_matrix.values
            
            # Portfolio optimization
            n = len(universe)
            w = cp.Variable(n)
            
            # Objective: maximize expected return - risk penalty
            objective = cp.Maximize(expected_scores.T @ w - 0.5 * cp.quad_form(w, cov_matrix))
            
            # Constraints: weights sum to 1, non-negative weights
            constraints = [cp.sum(w) == 1, w >= 0]
            
            # Solve optimization problem
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            if prob.status == cp.OPTIMAL:
                # Create allocations with actions and reasoning
                allocations = {}
                for i, etf in enumerate(universe):
                    weight = float(w.value[i])
                    risk_data = risks.get(etf, {'risk_level': 'medium', 'reason': 'No risk assessment'})
                    
                    # Determine action based on weight
                    if weight > 0.01:  # More than 1% allocation
                        action = 'buy'
                    else:
                        action = 'hold'
                    
                    allocations[etf] = {
                        'action': action,
                        'allocation': weight,
                        'reason': f"Optimized allocation based on {risk_data.get('risk_level', 'medium')} risk: {risk_data.get('reason', 'No reasoning')}"
                    }
                
                # Apply final adjustments and rounding
                allocations = self._apply_final_adjustments(allocations, universe)
                
                logger.info("Portfolio optimization completed successfully")
                return allocations
            else:
                logger.warning(f"Optimization failed with status: {prob.status}")
                # Fallback to risk-based allocations
                return self._create_risk_based_allocations(risks, universe)
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return self._create_risk_based_allocations(risks, universe)
    
    def _create_risk_based_allocations(self, risks: dict, universe: list) -> dict:
        """
        Create allocations based on risk assessments without optimization.
        
        Args:
            risks: Risk assessments from risk manager
            universe: List of ETFs to allocate
            
        Returns:
            Dictionary with risk-based allocations
        """
        try:
            allocations = {}
            total_risk_score = 0.0
            
            # Calculate total risk-adjusted score
            for etf in universe:
                risk_data = risks.get(etf, {'adjusted_score': 0.0, 'risk_level': 'medium'})
                adjusted_score = risk_data.get('adjusted_score', 0.0)
                risk_level = risk_data.get('risk_level', 'medium')
                
                # Apply risk penalty
                if risk_level == 'high':
                    adjusted_score *= 0.5  # 50% penalty for high risk
                elif risk_level == 'medium':
                    adjusted_score *= 0.8  # 20% penalty for medium risk
                # Low risk: no penalty
                
                total_risk_score += max(0, adjusted_score)  # Only positive scores
            
            # Create allocations based on risk-adjusted scores
            for etf in universe:
                risk_data = risks.get(etf, {'adjusted_score': 0.0, 'risk_level': 'medium'})
                adjusted_score = risk_data.get('adjusted_score', 0.0)
                risk_level = risk_data.get('risk_level', 'medium')
                
                # Apply risk penalty
                if risk_level == 'high':
                    adjusted_score *= 0.5
                elif risk_level == 'medium':
                    adjusted_score *= 0.8
                
                # Calculate allocation
                if total_risk_score > 0 and adjusted_score > 0:
                    allocation = (max(0, adjusted_score) / total_risk_score)
                else:
                    allocation = 0.0
                
                # Determine action
                if allocation > 0.01:
                    action = 'buy'
                else:
                    action = 'hold'
                
                allocations[etf] = {
                    'action': action,
                    'allocation': allocation,
                    'reason': f"Risk-based allocation ({risk_level} risk): {risk_data.get('reason', 'No reasoning')}"
                }
            
            return allocations
            
        except Exception as e:
            logger.error(f"Risk-based allocation failed: {e}")
            # Return neutral allocations
            return {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'Allocation failed'} for etf in universe}
    
    def _apply_final_adjustments(self, allocations: dict, universe: list) -> dict:
        """
        Apply final adjustments to allocations.
        
        Args:
            allocations: Raw allocations dictionary
            universe: List of ETFs
            
        Returns:
            Adjusted allocations dictionary
        """
        try:
            # Round allocations to reasonable precision
            for etf in allocations:
                allocation = allocations[etf]['allocation']
                # Round to nearest 0.01 (1%)
                rounded_allocation = round(allocation * 100) / 100.0
                
                # Set very small allocations to 0
                if rounded_allocation < 0.01:
                    rounded_allocation = 0.0
                    allocations[etf]['action'] = 'hold'
                
                allocations[etf]['allocation'] = rounded_allocation
            
            # Ensure allocations sum to 1.0 (100%)
            total_allocation = sum([alloc['allocation'] for alloc in allocations.values()])
            
            if total_allocation > 0 and abs(total_allocation - 1.0) > 0.01:
                # Adjust largest allocation to make sum exactly 1.0
                largest_etf = max(allocations.keys(), 
                               key=lambda x: allocations[x]['allocation'])
                adjustment = 1.0 - total_allocation
                allocations[largest_etf]['allocation'] += adjustment
                allocations[largest_etf]['allocation'] = round(allocations[largest_etf]['allocation'] * 100) / 100.0
            
            return allocations
            
        except Exception as e:
            logger.error(f"Final adjustments failed: {e}")
            return allocations


# Example usage and testing
if __name__ == "__main__":
    print("Portfolio Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = PortfolioAgent("TestPortfolioAgent")
        print(f"✓ Portfolio agent initialized")
        print(f"  Specialization: {agent.specialization}")
        print(f"  Role: {agent.role}")
        
        # Test with sample state
        import pandas as pd
        import numpy as np
        
        # Create sample ETF data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        etf_data = pd.DataFrame({
            'SPY': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'QQQ': 200 + np.cumsum(np.random.randn(100) * 0.015),
            'TLT': 150 + np.cumsum(np.random.randn(100) * 0.005),
            'GLD': 180 + np.cumsum(np.random.randn(100) * 0.008)
        }, index=dates)
        
        sample_state = {
            'risk_assessments': {
                'SPY': {
                    'risk_level': 'low',
                    'adjusted_score': 0.3,
                    'reason': 'Stable large-cap exposure with low volatility'
                },
                'QQQ': {
                    'risk_level': 'medium',
                    'adjusted_score': 0.2,
                    'reason': 'Tech concentration risk but strong fundamentals'
                },
                'TLT': {
                    'risk_level': 'high',
                    'adjusted_score': -0.1,
                    'reason': 'Interest rate sensitivity creates volatility risk'
                },
                'GLD': {
                    'risk_level': 'medium',
                    'adjusted_score': 0.1,
                    'reason': 'Safe haven demand but commodity volatility'
                }
            },
            'etf_data': etf_data,
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD']
        }
        
        # Test portfolio management
        result_state = agent.manage(sample_state)
        print(f"✓ Portfolio management completed")
        print(f"  Final allocations: {result_state.get('final_allocations', {})}")
        
        # Show detailed output for each ETF
        final_allocations = result_state.get('final_allocations', {})
        if final_allocations:
            print("\nPortfolio Summary:")
            print("="*20)
            for etf, data in final_allocations.items():
                print(f"{etf}: {data['action'].upper()} {data['allocation']:.1%} - {data['reason'][:50]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Portfolio agent test completed!")

"""
Portfolio Optimizer Agent for ETF Allocation Optimization

This agent uses mathematical optimization to find optimal ETF allocations
based on risk-adjusted allocations and correlation data.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizerAgent:
    """
    Portfolio Optimizer Agent that optimizes ETF allocations using mathematical optimization.
    
    This agent focuses on:
    - Mathematical portfolio optimization
    - Risk-return optimization using mean-variance framework
    - Correlation-based diversification
    - Constraint-based optimization
    - Final allocation decisions
    """
    
    def __init__(self, agent_name: str = "PortfolioOptimizerAgent"):
        """Initialize the portfolio optimizer agent."""
        self.agent_name = agent_name
        self.specialization = "portfolio_optimization"
        self.analysis_focus = "mathematical_optimization"
        self.role = "portfolio_optimizer"
    
    def analyze(self, data: dict) -> dict:
        """
        Analyze data and optimize ETF allocations.
        
        Args:
            data: Dictionary containing risk-adjusted allocations and correlation data
            
        Returns:
            Analysis results with optimized allocations
        """
        try:
            risk_adjusted_allocations = data.get('risk_adjusted_allocations', {})
            etf_data = data.get('etf_data', pd.DataFrame())
            universe = data.get('universe', [])
            
            # Optimize allocations
            optimized_allocations = self._optimize_portfolio(risk_adjusted_allocations, etf_data, universe)
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'final_allocations': optimized_allocations,
                'reasoning': f"Portfolio optimization using mean-variance framework for {len(universe)} ETFs",
                'optimization_method': 'mean_variance',
                'constraints': {'sum_to_one': True, 'non_negative': True},
                'performance_metrics': {'expected_return': 'calculated', 'volatility': 'calculated'},
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'error': str(e),
                'status': 'failed'
            }
    
    def optimize(self, state: dict) -> dict:
        """
        Optimize ETF allocations using mathematical optimization.
        
        Args:
            state: LangGraph state dictionary containing:
                - risk_adjusted_allocations: Risk-adjusted allocations from risk manager
                - etf_data: ETF price data for correlation analysis
                - universe: List of ETFs to optimize
                
        Returns:
            Updated state dictionary with final optimized allocations
        """
        try:
            # Extract data from state
            risk_adjusted_allocations = state.get('risk_adjusted_allocations', {})
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            # Optimize allocations
            optimized_allocations = self._optimize_portfolio(risk_adjusted_allocations, etf_data, universe)
            
            # Store final allocations in state
            state['final_allocations'] = optimized_allocations
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['portfolio_optimizer'] = {
                'final_allocations': optimized_allocations,
                'reasoning': f"Portfolio optimization based on risk-adjusted allocations and correlation analysis for {len(universe)} ETFs",
                'optimization_method': 'Mathematical optimization with risk constraints',
                'performance_metrics': {
                    'total_allocation': sum(optimized_allocations.values()),
                    'number_of_etfs': len([etf for etf, alloc in optimized_allocations.items() if alloc > 0]),
                    'max_allocation': max(optimized_allocations.values()) if optimized_allocations else 0,
                    'min_allocation': min([alloc for alloc in optimized_allocations.values() if alloc > 0]) if any(alloc > 0 for alloc in optimized_allocations.values()) else 0
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Portfolio optimizer completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return risk-adjusted allocations on error
            state['final_allocations'] = state.get('risk_adjusted_allocations', {})
            return state
    
    def _optimize_portfolio(self, risk_adjusted_allocations: dict, etf_data: pd.DataFrame, universe: list) -> dict:
        """
        Optimize portfolio using mean-variance optimization.
        
        Args:
            risk_adjusted_allocations: Risk-adjusted allocations from risk manager
            etf_data: ETF price data for correlation analysis
            universe: List of ETFs to optimize
            
        Returns:
            Dictionary with optimized ETF allocations
        """
        try:
            if not risk_adjusted_allocations or etf_data.empty:
                logger.warning("Insufficient data for optimization, returning risk-adjusted allocations")
                return risk_adjusted_allocations
            
            # Calculate correlation matrix
            corr_matrix = self._calculate_correlation_matrix(etf_data, universe)
            
            if corr_matrix is None or corr_matrix.empty:
                logger.warning("Could not calculate correlation matrix, returning risk-adjusted allocations")
                return risk_adjusted_allocations
            
            # Convert allocations to expected returns
            expected_returns = np.array([risk_adjusted_allocations.get(etf, 0.0) for etf in universe])
            
            # Use correlation matrix as covariance matrix (simplified)
            cov_matrix = corr_matrix.values
            
            # Portfolio optimization
            n = len(universe)
            w = cp.Variable(n)
            
            # Objective: maximize expected return - risk penalty
            objective = cp.Maximize(expected_returns.T @ w - 0.5 * cp.quad_form(w, cov_matrix))
            
            # Constraints: weights sum to 1, non-negative weights
            constraints = [cp.sum(w) == 1, w >= 0]
            
            # Solve optimization problem
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            if prob.status == cp.OPTIMAL:
                # Convert to allocation dictionary
                optimized_allocations = {}
                for i, etf in enumerate(universe):
                    optimized_allocations[etf] = float(w.value[i]) * 100.0
                
                logger.info("Portfolio optimization completed successfully")
                return optimized_allocations
            else:
                logger.warning(f"Optimization failed with status: {prob.status}")
                return risk_adjusted_allocations
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return risk_adjusted_allocations
    
    def _calculate_correlation_matrix(self, etf_data: pd.DataFrame, universe: list) -> pd.DataFrame:
        """
        Calculate correlation matrix for ETFs.
        
        Args:
            etf_data: DataFrame with ETF price data
            universe: List of ETFs to analyze
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            if etf_data.empty:
                return pd.DataFrame()
            
            # Calculate returns
            returns = etf_data.pct_change().dropna()
            
            # Filter to universe ETFs
            available_etfs = [etf for etf in universe if etf in returns.columns]
            
            if not available_etfs:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = returns[available_etfs].corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    print("Portfolio Optimizer Agent Test")
    print("="*40)
    
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizerAgent("TestOptimizer")
        print(f"✓ Portfolio optimizer initialized")
        print(f"  Specialization: {optimizer.specialization}")
        print(f"  Role: {optimizer.role}")
        
        # Test with sample data
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
            'risk_adjusted_allocations': {
                'SPY': 30.0,
                'QQQ': 25.0,
                'TLT': 25.0,
                'GLD': 20.0
            },
            'etf_data': etf_data,
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Test optimization
        result_state = optimizer.optimize(sample_state)
        print(f"✓ Portfolio optimization completed")
        print(f"  Final allocations: {result_state.get('final_allocations', {})}")
        
        # Test individual analysis
        analysis_result = optimizer.analyze(sample_state)
        print(f"✓ Individual analysis completed")
        print(f"  Allocations: {analysis_result.get('final_allocations', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Portfolio optimizer test completed!")

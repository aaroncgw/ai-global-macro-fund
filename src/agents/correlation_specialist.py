"""
Correlation Specialist Agent

This agent analyzes ETF correlations and suggests diversification scores
based on correlation patterns and portfolio optimization principles.
"""

from src.agents.base_agent import BaseAgent
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class CorrelationSpecialistAgent(BaseAgent):
    """
    Correlation Specialist Agent that analyzes ETF correlations and diversification.
    
    This agent focuses on:
    - ETF correlation analysis
    - Diversification opportunities
    - Portfolio optimization
    - Risk reduction through correlation
    - Scoring ETFs from -1 (sell) to 1 (buy) based on diversification value
    """
    
    def __init__(self, agent_name: str = "CorrelationSpecialistAgent"):
        """Initialize the correlation specialist agent."""
        super().__init__(agent_name)
        self.specialization = "correlation_analysis"
        self.analysis_focus = "diversification"
    
    def analyze(self, state: dict) -> dict:
        """
        Analyze ETF correlations and score for diversification.
        
        Args:
            state: LangGraph state dictionary containing:
                - etf_data: ETF price data
                - universe: List of ETFs to analyze
                - analyst_scores: Dictionary to store scores
                
        Returns:
            Updated state dictionary with correlation specialist scores
        """
        try:
            # Extract data from state
            etf_data = state.get('etf_data', pd.DataFrame())
            universe = state.get('universe', [])
            
            # Ensure analyst_scores exists in state
            if 'analyst_scores' not in state:
                state['analyst_scores'] = {}
            
            # Calculate correlation matrix
            corr_matrix = self._calculate_correlation_matrix(etf_data, universe)
            
            # Create analysis prompt
            prompt = self._create_correlation_analysis_prompt(corr_matrix, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse scores from response
            scores = self._parse_scores(response, universe)
            
            # Store scores in state
            state['analyst_scores']['correlation'] = scores
            
            logger.info(f"Correlation analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: 0.0 for etf in state.get('universe', [])}
            state['analyst_scores']['correlation'] = neutral_scores
            return state
    
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
                logger.warning("No ETF data available for correlation analysis")
                return pd.DataFrame()
            
            # Calculate returns
            returns = etf_data.pct_change().dropna()
            
            # Filter to universe ETFs
            available_etfs = [etf for etf in universe if etf in returns.columns]
            
            if not available_etfs:
                logger.warning("No ETFs from universe found in data")
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = returns[available_etfs].corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()
    
    def _create_correlation_analysis_prompt(self, corr_matrix: pd.DataFrame, universe: list) -> str:
        """
        Create a prompt for correlation analysis.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            universe: List of ETFs to analyze
            
        Returns:
            Formatted prompt string
        """
        # Format correlation matrix
        corr_summary = self._format_correlation_matrix(corr_matrix)
        
        prompt = f"""
        As a correlation specialist, analyze the following ETF correlations and score each ETF from -1 (strong sell) to 1 (strong buy) based on diversification value:
        
        ETF CORRELATION MATRIX:
        {corr_summary}
        
        ANALYSIS FRAMEWORK:
        1. High Correlation (0.7+): ETFs that move together - consider reducing exposure
        2. Low Correlation (0.3-): ETFs that provide diversification - consider increasing exposure
        3. Negative Correlation (-0.3-): ETFs that move opposite - excellent for hedging
        4. Portfolio Balance: Which ETFs add unique risk/return profiles?
        5. Sector/Region Diversification: How do correlations vary across asset classes?
        
        SCORING CRITERIA:
        - Score each ETF from -1.0 (strong sell) to 1.0 (strong buy)
        - Favor ETFs with low correlation to others (diversification value)
        - Penalize ETFs with very high correlation (redundancy)
        - Consider both absolute correlation and relative diversification benefit
        - Focus on portfolio optimization and risk reduction
        
        ETFs TO SCORE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores:
        {{"SPY": 0.2, "QQQ": -0.1, "TLT": 0.4, ...}}
        """
        return prompt
    
    def _format_correlation_matrix(self, corr_matrix: pd.DataFrame) -> str:
        """Format correlation matrix for the prompt."""
        if corr_matrix.empty:
            return "No correlation data available"
        
        try:
            # Convert to dictionary for JSON serialization
            corr_dict = corr_matrix.to_dict()
            
            # Format as readable table
            formatted = []
            formatted.append("Correlation Matrix (values between -1 and 1):")
            formatted.append("")
            
            # Add header
            etfs = list(corr_dict.keys())
            header = "ETF".ljust(8) + "".join([etf.ljust(8) for etf in etfs])
            formatted.append(header)
            formatted.append("-" * len(header))
            
            # Add rows
            for etf1 in etfs:
                row = etf1.ljust(8)
                for etf2 in etfs:
                    if etf1 in corr_dict and etf2 in corr_dict[etf1]:
                        corr_value = corr_dict[etf1][etf2]
                        row += f"{corr_value:.2f}".ljust(8)
                    else:
                        row += "N/A".ljust(8)
                formatted.append(row)
            
            # Add summary statistics
            formatted.append("")
            formatted.append("Key Correlations:")
            for etf1 in etfs:
                for etf2 in etfs:
                    if etf1 != etf2 and etf1 in corr_dict and etf2 in corr_dict[etf1]:
                        corr_value = corr_dict[etf1][etf2]
                        if abs(corr_value) > 0.7:  # High correlation
                            formatted.append(f"- {etf1} vs {etf2}: {corr_value:.2f} (High correlation)")
                        elif abs(corr_value) < 0.3:  # Low correlation
                            formatted.append(f"- {etf1} vs {etf2}: {corr_value:.2f} (Low correlation)")
            
            return '\n'.join(formatted)
            
        except Exception as e:
            logger.error(f"Failed to format correlation matrix: {e}")
            return "Error formatting correlation data"
    
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
    print("Correlation Specialist Agent Test")
    print("="*40)
    
    try:
        # Initialize agent
        agent = CorrelationSpecialistAgent("TestCorrelationSpecialist")
        print(f"✓ Correlation specialist agent initialized")
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
            'QQQ': 200 + np.cumsum(np.random.randn(100) * 0.015),
            'TLT': 150 + np.cumsum(np.random.randn(100) * 0.005),
            'GLD': 180 + np.cumsum(np.random.randn(100) * 0.008)
        }, index=dates)
        
        sample_state = {
            'etf_data': etf_data,
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'analyst_scores': {}
        }
        
        # Test analysis
        result_state = agent.analyze(sample_state)
        print(f"✓ Analysis completed")
        print(f"  Scores: {result_state.get('analyst_scores', {}).get('correlation', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Correlation specialist test completed!")

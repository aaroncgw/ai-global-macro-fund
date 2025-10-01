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
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['correlation_specialist'] = {
                'scores': scores,
                'reasoning': f"Correlation analysis based on {len(universe)} ETFs and correlation matrix",
                'key_factors': ['correlation_matrix', 'diversification_benefits', 'portfolio_balance'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
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
        
        num_etfs = len(corr_matrix) if not corr_matrix.empty else len(universe)
        
        prompt = f"""
        As a correlation specialist, analyze the ETF correlations and score each ETF from -1 (strong sell) to 1 (strong buy) based on diversification value.
        
        You are analyzing {num_etfs} ETFs. The correlation data below shows which ETFs provide diversification vs redundancy.
        
        ETF CORRELATION ANALYSIS:
        {corr_summary}
        
        ANALYSIS FRAMEWORK:
        1. **Diversification Value (POSITIVE scores):**
           - ETFs with LOW average correlation (<0.4) add unique exposure
           - Asset classes that are uncorrelated with equities (bonds, commodities)
           - Currency ETFs that hedge regional risks
           - Score range: +0.5 to +1.0 for best diversifiers
        
        2. **Redundancy Risk (NEGATIVE scores):**
           - ETFs with HIGH average correlation (>0.7) are redundant
           - Asset classes that move in lockstep (all equities together)
           - Regional ETFs highly correlated with global indices
           - Score range: -0.5 to -1.0 for most redundant
        
        3. **Moderate Diversification (NEUTRAL scores):**
           - ETFs with moderate correlation (0.4-0.7)
           - Score range: -0.3 to +0.3
        
        SCORING STRATEGY FOR {num_etfs} ETFs:
        - Review the "AVERAGE CORRELATION PER ETF" section
        - ETFs with avg correlation < 0.3 → Score +0.7 to +1.0 (excellent diversifiers)
        - ETFs with avg correlation 0.3-0.5 → Score +0.2 to +0.5 (good diversifiers)
        - ETFs with avg correlation 0.5-0.7 → Score -0.2 to +0.2 (moderate)
        - ETFs with avg correlation > 0.7 → Score -0.5 to -1.0 (redundant)
        
        - Bonds (TLT, IEF, BND) typically have LOW correlation with equities → POSITIVE scores
        - Commodities (GLD, SLV, USO) often have LOW correlation → POSITIVE scores
        - Currency ETFs (UUP, FXE, FXY) can provide unique diversification → POSITIVE scores
        - Multiple equity ETFs from same region → NEGATIVE scores for redundancy
        
        IMPORTANT: Differentiate scores meaningfully. NOT all ETFs should be 0.0!
        Use the full range from -1.0 to +1.0 based on correlation data.
        
        ETFs TO SCORE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores (use full range -1 to +1):
        {{"SPY": -0.3, "QQQ": -0.4, "TLT": 0.8, "GLD": 0.7, "UUP": 0.5, ...}}
        """
        return prompt
    
    def _format_correlation_matrix(self, corr_matrix: pd.DataFrame) -> str:
        """
        Format correlation matrix for the prompt.
        For large universes (>10 ETFs), use summary statistics instead of full matrix.
        """
        if corr_matrix.empty:
            return "No correlation data available"
        
        try:
            num_etfs = len(corr_matrix)
            
            # For small universes (<= 10 ETFs), show full matrix
            if num_etfs <= 10:
                return self._format_full_correlation_matrix(corr_matrix)
            
            # For large universes (> 10 ETFs), use smart summary
            return self._format_correlation_summary(corr_matrix)
            
        except Exception as e:
            logger.error(f"Failed to format correlation matrix: {e}")
            return "Error formatting correlation data"
    
    def _format_full_correlation_matrix(self, corr_matrix: pd.DataFrame) -> str:
        """Format full correlation matrix for small universes."""
        corr_dict = corr_matrix.to_dict()
        formatted = []
        formatted.append("Correlation Matrix (values between -1 and 1):")
        formatted.append("")
        
        # Add header
        etfs = [str(etf) for etf in corr_dict.keys()]
        header = "ETF".ljust(8) + "".join([etf.ljust(8) for etf in etfs])
        formatted.append(header)
        formatted.append("-" * len(header))
        
        # Add rows
        for etf1 in etfs:
            row = str(etf1).ljust(8)
            for etf2 in etfs:
                if etf1 in corr_dict and etf2 in corr_dict[etf1]:
                    corr_value = corr_dict[etf1][etf2]
                    row += f"{corr_value:.2f}".ljust(8)
                else:
                    row += "N/A".ljust(8)
            formatted.append(row)
        
        return '\n'.join(formatted)
    
    def _format_correlation_summary(self, corr_matrix: pd.DataFrame) -> str:
        """Format correlation summary for large universes using smart analytics."""
        formatted = []
        formatted.append(f"=== CORRELATION ANALYSIS FOR {len(corr_matrix)} ETFs ===\n")
        
        # 1. Calculate average correlation for each ETF
        formatted.append("AVERAGE CORRELATION PER ETF (lower = better diversification):")
        avg_corr = {}
        for etf in corr_matrix.columns:
            # Calculate average correlation with all other ETFs
            other_etfs = [e for e in corr_matrix.columns if e != etf]
            avg_corr[etf] = corr_matrix.loc[etf, other_etfs].mean()
        
        # Sort by average correlation
        sorted_etfs = sorted(avg_corr.items(), key=lambda x: x[1])
        
        for etf, avg in sorted_etfs:
            diversification = "🟢 High Diversification" if avg < 0.3 else "🟡 Moderate" if avg < 0.6 else "🔴 Low Diversification"
            formatted.append(f"  {etf}: {avg:.3f} {diversification}")
        
        # 2. Identify highly correlated pairs (redundancy risk)
        formatted.append(f"\nHIGH CORRELATION PAIRS (>0.8) - Redundancy Risk:")
        high_corr_pairs = []
        for i, etf1 in enumerate(corr_matrix.columns):
            for etf2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[etf1, etf2]
                if corr_val > 0.8:
                    high_corr_pairs.append(f"  {etf1} - {etf2}: {corr_val:.3f}")
        
        if high_corr_pairs:
            formatted.extend(high_corr_pairs[:10])  # Show top 10
            if len(high_corr_pairs) > 10:
                formatted.append(f"  ... and {len(high_corr_pairs) - 10} more pairs")
        else:
            formatted.append("  None found - Good diversification")
        
        # 3. Identify low/negative correlation pairs (diversification opportunities)
        formatted.append(f"\nLOW CORRELATION PAIRS (<0.3) - Diversification Benefits:")
        low_corr_pairs = []
        for i, etf1 in enumerate(corr_matrix.columns):
            for etf2 in corr_matrix.columns[i+1:]:
                corr_val = corr_matrix.loc[etf1, etf2]
                if corr_val < 0.3:
                    low_corr_pairs.append(f"  {etf1} - {etf2}: {corr_val:.3f}")
        
        if low_corr_pairs:
            formatted.extend(low_corr_pairs[:10])  # Show top 10
            if len(low_corr_pairs) > 10:
                formatted.append(f"  ... and {len(low_corr_pairs) - 10} more pairs")
        else:
            formatted.append("  None found - Everything is correlated")
        
        # 4. Asset class groupings
        formatted.append(f"\nASSET CLASS CORRELATION SUMMARY:")
        
        # Group ETFs by type
        equity_etfs = [etf for etf in corr_matrix.columns if etf in ['SPY', 'QQQ', 'VEU', 'VWO', 'VGK', 'VPL', 'ACWI', 
                                                                       'EWJ', 'EWG', 'EWU', 'EWA', 'EWC', 'EWZ', 'INDA', 'FXI', 'EZA', 'TUR', 'RSX', 'EWW']]
        bond_etfs = [etf for etf in corr_matrix.columns if etf in ['TLT', 'IEF', 'BND', 'TIP', 'LQD', 'HYG', 'EMB', 'PCY']]
        commodity_etfs = [etf for etf in corr_matrix.columns if etf in ['GLD', 'SLV', 'USO', 'UNG', 'DBC', 'CORN', 'WEAT', 'DBA', 'PDBC', 'GSG']]
        currency_etfs = [etf for etf in corr_matrix.columns if etf in ['UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA', 'FXF', 'CYB']]
        
        # Calculate intra-group and inter-group correlations
        if equity_etfs:
            eq_avg = corr_matrix.loc[equity_etfs, equity_etfs].mean().mean()
            formatted.append(f"  Equities (intra-group avg): {eq_avg:.3f}")
        
        if bond_etfs:
            bond_avg = corr_matrix.loc[bond_etfs, bond_etfs].mean().mean()
            formatted.append(f"  Bonds (intra-group avg): {bond_avg:.3f}")
        
        if commodity_etfs:
            comm_avg = corr_matrix.loc[commodity_etfs, commodity_etfs].mean().mean()
            formatted.append(f"  Commodities (intra-group avg): {comm_avg:.3f}")
        
        if equity_etfs and bond_etfs:
            eq_bond = corr_matrix.loc[equity_etfs, bond_etfs].mean().mean()
            formatted.append(f"  Equities vs Bonds: {eq_bond:.3f} (diversification benefit)")
        
        if equity_etfs and commodity_etfs:
            eq_comm = corr_matrix.loc[equity_etfs, commodity_etfs].mean().mean()
            formatted.append(f"  Equities vs Commodities: {eq_comm:.3f}")
        
        # 5. Key insights for scoring
        formatted.append(f"\n=== SCORING GUIDANCE ===")
        formatted.append(f"ETFs with LOW average correlation (<0.4) should receive POSITIVE scores (good diversifiers)")
        formatted.append(f"ETFs with HIGH average correlation (>0.7) should receive NEGATIVE scores (redundancy)")
        formatted.append(f"Bonds and Commodities typically provide best diversification vs equities")
        
        return '\n'.join(formatted)
    
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

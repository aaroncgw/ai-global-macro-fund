"""
Example Agent Implementation

This module demonstrates how to extend the BaseAgent class
to create specific agent implementations for the global macro system.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent


class ExampleMacroAgent(BaseAgent):
    """
    Example implementation of a macro agent that extends BaseAgent.
    
    This demonstrates how to create specialized agents for specific
    macro analysis tasks while maintaining LLM flexibility.
    """
    
    def __init__(self, agent_name: str = "ExampleMacroAgent"):
        """Initialize the example macro agent."""
        super().__init__(agent_name)
        self.specialization = "example_macro_analysis"
        self.analysis_focus = "trend_analysis"
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform specialized macro analysis.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Analysis results with specialized insights
        """
        try:
            # Extract data specific to this agent's focus
            etfs = data.get('etfs', [])
            macro_indicators = data.get('macro_indicators', {})
            etf_data = data.get('etf_data', {})
            
            # Create specialized analysis prompt
            prompt = self._create_specialized_prompt(etfs, macro_indicators, etf_data)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse and structure the response
            analysis_result = {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'analysis_focus': self.analysis_focus,
                'etfs_analyzed': etfs,
                'indicators_used': list(macro_indicators.keys()),
                'llm_response': response,
                'timestamp': data.get('timestamp', 'unknown'),
                'provider_info': self.get_provider_info()
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'error': str(e),
                'status': 'failed'
            }
    
    def _create_specialized_prompt(self, etfs: list, indicators: dict, etf_data: dict) -> str:
        """Create a specialized prompt for this agent's analysis."""
        prompt = f"""
        As a specialized macro analyst focusing on trend analysis, analyze the following:
        
        ETFs: {', '.join(etfs)}
        Macro Indicators: {list(indicators.keys())}
        
        Provide a focused analysis on:
        1. Trend identification and momentum
        2. Breakout opportunities
        3. Risk management considerations
        4. Specific allocation recommendations
        
        Focus on technical and fundamental trend analysis.
        """
        return prompt


class RiskAnalysisAgent(BaseAgent):
    """
    Example risk analysis agent that extends BaseAgent.
    
    This demonstrates how to create agents with different specializations
    while using the same base LLM infrastructure.
    """
    
    def __init__(self, agent_name: str = "RiskAnalysisAgent"):
        """Initialize the risk analysis agent."""
        super().__init__(agent_name)
        self.specialization = "risk_analysis"
        self.analysis_focus = "risk_assessment"
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk analysis on the provided data."""
        try:
            etfs = data.get('etfs', [])
            macro_indicators = data.get('macro_indicators', {})
            etf_returns = data.get('etf_returns', {})
            
            # Create risk-focused prompt
            prompt = self._create_risk_prompt(etfs, macro_indicators, etf_returns)
            
            # Get LLM response
            response = self.llm(prompt)
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'analysis_focus': self.analysis_focus,
                'etfs_analyzed': etfs,
                'risk_assessment': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'error': str(e),
                'status': 'failed'
            }
    
    def _create_risk_prompt(self, etfs: list, indicators: dict, returns: dict) -> str:
        """Create a risk-focused analysis prompt."""
        prompt = f"""
        As a risk analyst, assess the following portfolio and macro environment:
        
        ETFs: {', '.join(etfs)}
        Macro Indicators: {list(indicators.keys())}
        
        Provide a comprehensive risk assessment including:
        1. Portfolio risk metrics
        2. Macro risk factors
        3. Correlation analysis
        4. Risk mitigation strategies
        5. Stress testing scenarios
        
        Focus on downside protection and risk management.
        """
        return prompt


# Example usage
if __name__ == "__main__":
    print("Example Agent Test Suite")
    print("="*50)
    
    # Test example macro agent
    try:
        macro_agent = ExampleMacroAgent("TestMacroAgent")
        print(f"✓ Example macro agent initialized")
        print(f"  Specialization: {macro_agent.specialization}")
        print(f"  Provider: {macro_agent.get_provider_info()['provider']}")
        
        # Test risk analysis agent
        risk_agent = RiskAnalysisAgent("TestRiskAgent")
        print(f"✓ Risk analysis agent initialized")
        print(f"  Specialization: {risk_agent.specialization}")
        print(f"  Provider: {risk_agent.get_provider_info()['provider']}")
        
        # Test analysis with sample data
        sample_data = {
            'etfs': ['SPY', 'QQQ', 'TLT'],
            'macro_indicators': {'CPIAUCSL': {'latest_value': 300.0}},
            'etf_data': {},
            'etf_returns': {},
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Test macro analysis
        macro_result = macro_agent.analyze(sample_data)
        print(f"✓ Macro analysis completed")
        print(f"  Analysis focus: {macro_result.get('analysis_focus', 'unknown')}")
        
        # Test risk analysis
        risk_result = risk_agent.analyze(sample_data)
        print(f"✓ Risk analysis completed")
        print(f"  Analysis focus: {risk_result.get('analysis_focus', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Example agent test completed!")

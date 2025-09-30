"""
Risk Manager Agent for ETF Allocation Risk Assessment
"""

from src.agents.base_agent import BaseAgent
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """Risk Manager Agent that assesses and adjusts ETF allocations for risk."""
    
    def __init__(self, agent_name: str = "RiskManagerAgent"):
        super().__init__(agent_name)
        self.specialization = "risk_management"
        self.analysis_focus = "risk_assessment"
        self.role = "risk_manager"
    
    def analyze(self, data: dict) -> dict:
        """Analyze data and assess risk-adjusted allocations."""
        try:
            proposed_allocations = data.get('proposed_allocations', {})
            macro_data = data.get('macro_data', {})
            risk_factors = data.get('risk_factors', {})
            
            prompt = self._create_risk_assessment_prompt(proposed_allocations, macro_data, risk_factors)
            response = self.llm(prompt)
            risk_adjusted_allocations = self._parse_allocations(response, list(proposed_allocations.keys()))
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'risk_adjusted_allocations': risk_adjusted_allocations,
                'reasoning': f"Risk assessment based on {len(proposed_allocations)} proposed allocations and macro data",
                'risk_factors': list(risk_factors.keys()) if risk_factors else ['No specific risk factors identified'],
                'adjustments': {etf: risk_adjusted_allocations.get(etf, 0) - proposed_allocations.get(etf, 0) for etf in proposed_allocations.keys()},
                'llm_response': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Risk manager analysis failed: {e}")
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'role': self.role,
                'error': str(e),
                'status': 'failed'
            }
    
    def assess(self, state: dict) -> dict:
        """Assess and adjust ETF allocations for risk."""
        try:
            proposed_allocations = state.get('proposed_allocations', {})
            macro_data = state.get('macro_data', {})
            risk_factors = state.get('risk_factors', {})
            
            prompt = self._create_risk_assessment_prompt(proposed_allocations, macro_data, risk_factors)
            response = self.llm(prompt)
            risk_adjusted_allocations = self._parse_allocations(response, list(proposed_allocations.keys()))
            
            state['risk_adjusted_allocations'] = risk_adjusted_allocations
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['risk_manager'] = {
                'risk_adjusted_allocations': risk_adjusted_allocations,
                'reasoning': f"Risk assessment based on {len(proposed_allocations)} proposed allocations and macro data",
                'risk_factors': list(risk_factors.keys()) if risk_factors else ['No specific risk factors identified'],
                'adjustments': {etf: risk_adjusted_allocations.get(etf, 0) - proposed_allocations.get(etf, 0) for etf in proposed_allocations.keys()},
                'llm_response': response,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Risk manager assessed allocations for {len(proposed_allocations)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"Risk manager assessment failed: {e}")
            state['risk_adjusted_allocations'] = state.get('proposed_allocations', {})
            return state
    
    def _create_risk_assessment_prompt(self, proposed_allocations: dict, macro_data: dict, risk_factors: dict) -> str:
        """Create a prompt for risk assessment and allocation adjustment."""
        allocations_summary = self._format_allocations(proposed_allocations)
        macro_summary = self._format_macro_data(macro_data)
        risk_summary = self._format_risk_factors(risk_factors)
        
        prompt = f"""
        As a risk manager, assess and adjust the following ETF allocations for macro risks and volatility:
        
        PROPOSED ALLOCATIONS:
        {allocations_summary}
        
        MACRO ECONOMIC DATA:
        {macro_summary}
        
        RISK FACTORS:
        {risk_summary}
        
        RISK ASSESSMENT REQUIREMENTS:
        1. Adjust allocations based on macro risk factors
        2. Cap volatile ETFs if inflation is high
        3. Reduce exposure to risky assets during economic uncertainty
        4. Increase defensive positioning during market stress
        5. Consider correlation risks and concentration
        6. Account for liquidity and market depth
        7. Balance risk and return objectives
        8. Provide specific reasons for adjustments
        
        Return ONLY a JSON object with risk-adjusted ETF allocations as percentages:
        {{"SPY": 20.0, "QQQ": 15.0, "TLT": 25.0, "GLD": 10.0, ...}}
        """
        return prompt
    
    def _format_allocations(self, allocations: dict) -> str:
        """Format proposed allocations for the prompt."""
        if not allocations:
            return "No proposed allocations available"
        
        formatted = []
        for etf, allocation in allocations.items():
            formatted.append(f"  {etf}: {allocation:.1f}%")
        
        return '\n'.join(formatted)
    
    def _format_macro_data(self, macro_data: dict) -> str:
        """Format macro data for the prompt."""
        if not macro_data:
            return "No macro data available"
        
        formatted = []
        for indicator, data in macro_data.items():
            if 'error' not in data and data.get('latest_value') is not None:
                latest_value = data.get('latest_value', 'N/A')
                trend = data.get('trend', 'unknown')
                formatted.append(f"- {indicator}: {latest_value} (trend: {trend})")
            else:
                formatted.append(f"- {indicator}: Error - {data.get('error', 'No data')}")
        
        return '\n'.join(formatted) if formatted else "No macro data available"
    
    def _format_risk_factors(self, risk_factors: dict) -> str:
        """Format risk factors for the prompt."""
        if not risk_factors:
            return "No additional risk factors specified"
        
        formatted = []
        for factor, value in risk_factors.items():
            formatted.append(f"- {factor}: {value}")
        
        return '\n'.join(formatted)
    
    def _parse_allocations(self, response: str, universe: list) -> dict:
        """Parse risk-adjusted ETF allocations from LLM response."""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                allocations = json.loads(json_str)
                
                validated_allocations = {}
                total_allocation = 0.0
                
                for etf in universe:
                    if etf in allocations:
                        allocation = float(allocations[etf])
                        allocation = max(0.0, allocation)
                        validated_allocations[etf] = allocation
                        total_allocation += allocation
                    else:
                        validated_allocations[etf] = 0.0
                
                if total_allocation > 0:
                    for etf in validated_allocations:
                        validated_allocations[etf] = (validated_allocations[etf] / total_allocation) * 100.0
                else:
                    equal_allocation = 100.0 / len(universe)
                    for etf in universe:
                        validated_allocations[etf] = equal_allocation
                
                return validated_allocations
            else:
                logger.warning("No JSON found in LLM response")
                equal_allocation = 100.0 / len(universe)
                return {etf: equal_allocation for etf in universe}
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse risk-adjusted allocations from response: {e}")
            equal_allocation = 100.0 / len(universe)
            return {etf: equal_allocation for etf in universe}


if __name__ == "__main__":
    print("Risk Manager Agent Test")
    print("="*30)
    
    try:
        risk_manager = RiskManagerAgent("TestRiskManager")
        print(f"✓ Risk manager agent initialized")
        print(f"  Specialization: {risk_manager.specialization}")
        print(f"  Provider: {risk_manager.get_provider_info()['provider']}")
        
        sample_state = {
            'proposed_allocations': {
                'SPY': 30.0,
                'QQQ': 25.0,
                'TLT': 20.0,
                'GLD': 15.0,
                'EWJ': 10.0
            },
            'macro_data': {
                'CPIAUCSL': {'latest_value': 300.0, 'trend': 'increasing'},
                'UNRATE': {'latest_value': 3.5, 'trend': 'stable'},
                'FEDFUNDS': {'latest_value': 5.25, 'trend': 'increasing'}
            },
            'risk_factors': {
                'volatility_regime': 'high',
                'geopolitical_risk': 'elevated',
                'liquidity_conditions': 'normal'
            }
        }
        
        result_state = risk_manager.assess(sample_state)
        print(f"✓ Risk assessment completed")
        print(f"  Risk-adjusted allocations: {result_state.get('risk_adjusted_allocations', {})}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*30)
    print("Risk manager test completed!")

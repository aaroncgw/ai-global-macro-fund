"""
Debate Researchers for Macro ETF Analysis

This module implements debate researchers that engage in structured debates
to analyze ETF opportunities and risks from different perspectives.
"""

from src.agents.base_agent import BaseAgent
from src.config import DEFAULT_CONFIG
import logging

logger = logging.getLogger(__name__)


class BullishMacroResearcher(BaseAgent):
    """
    Bullish Macro Researcher that argues for positive macro opportunities.
    
    This researcher focuses on:
    - Bullish macro trends and opportunities
    - Positive economic indicators
    - Growth opportunities in different asset classes
    - Inflation hedges and growth plays
    - Risk-on sentiment and market optimism
    """
    
    def __init__(self, agent_name: str = "BullishMacroResearcher"):
        """Initialize the bullish macro researcher."""
        super().__init__(agent_name)
        self.specialization = "bullish_macro_research"
        self.analysis_focus = "opportunities_and_growth"
        self.perspective = "bullish"
    
    def analyze(self, data: dict) -> dict:
        """
        Analyze data and return bullish research argument.
        
        Args:
            data: Dictionary containing analyst_scores and round_number
            
        Returns:
            Analysis results with bullish research argument
        """
        try:
            analyst_scores = data.get('analyst_scores', {})
            round_number = data.get('round_number', 1)
            
            # Create analysis prompt
            prompt = f"""
            As a bullish macro researcher, present compelling arguments for macro opportunities in Round {round_number}.
            
            ANALYST SCORES TO CONSIDER:
            {self._format_analyst_scores(analyst_scores)}
            
            BULLISH RESEARCH FOCUS:
            1. Identify the strongest macro opportunities based on analyst scores
            2. Highlight positive economic trends and catalysts
            3. Present growth opportunities in different asset classes
            4. Argue for inflation hedges and growth plays
            5. Emphasize risk-on sentiment and market optimism
            6. Counter any bearish arguments from previous rounds
            
            RESEARCH REQUIREMENTS:
            - Focus on ETFs with positive scores from analysts
            - Highlight macro trends supporting bullish thesis
            - Present specific economic data and indicators
            - Argue for portfolio positioning in growth assets
            - Address potential risks with bullish counterarguments
            
            Provide a comprehensive bullish research argument.
            """
            
            response = self.llm(prompt)
            logger.info(f"Bullish researcher completed Round {round_number}")
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'perspective': self.perspective,
                'round_number': round_number,
                'research_argument': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Bullish research failed in Round {round_number}: {e}")
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'perspective': self.perspective,
                'error': str(e),
                'status': 'failed'
            }
    
    def research_opportunities(self, analyst_scores: dict, round_number: int) -> str:
        """
        Research and present bullish macro opportunities.
        
        Args:
            analyst_scores: Dictionary of analyst scores
            round_number: Current debate round number
            
        Returns:
            Bullish research argument
        """
        try:
            prompt = f"""
            As a bullish macro researcher, present compelling arguments for macro opportunities in Round {round_number}.
            
            ANALYST SCORES TO CONSIDER:
            {self._format_analyst_scores(analyst_scores)}
            
            BULLISH RESEARCH FOCUS:
            1. Identify the strongest macro opportunities based on analyst scores
            2. Highlight positive economic trends and catalysts
            3. Present growth opportunities in different asset classes
            4. Argue for inflation hedges and growth plays
            5. Emphasize risk-on sentiment and market optimism
            6. Counter any bearish arguments from previous rounds
            
            RESEARCH REQUIREMENTS:
            - Focus on ETFs with positive scores from analysts
            - Highlight macro trends supporting bullish thesis
            - Present specific economic data and indicators
            - Argue for portfolio positioning in growth assets
            - Address potential risks with bullish counterarguments
            
            Provide a comprehensive bullish research argument.
            """
            
            response = self.llm(prompt)
            logger.info(f"Bullish researcher completed Round {round_number}")
            return response
            
        except Exception as e:
            logger.error(f"Bullish research failed in Round {round_number}: {e}")
            return f"Bullish research error: {str(e)}"
    
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


class BearishMacroResearcher(BaseAgent):
    """
    Bearish Macro Researcher that argues for macro risks and challenges.
    
    This researcher focuses on:
    - Bearish macro risks and challenges
    - Negative economic indicators
    - Downside risks in different asset classes
    - Deflation risks and market pessimism
    - Risk-off sentiment and defensive positioning
    """
    
    def __init__(self, agent_name: str = "BearishMacroResearcher"):
        """Initialize the bearish macro researcher."""
        super().__init__(agent_name)
        self.specialization = "bearish_macro_research"
        self.analysis_focus = "risks_and_challenges"
        self.perspective = "bearish"
    
    def analyze(self, data: dict) -> dict:
        """
        Analyze data and return bearish research argument.
        
        Args:
            data: Dictionary containing analyst_scores, bullish_argument, and round_number
            
        Returns:
            Analysis results with bearish research argument
        """
        try:
            analyst_scores = data.get('analyst_scores', {})
            bullish_argument = data.get('bullish_argument', '')
            round_number = data.get('round_number', 1)
            
            # Create analysis prompt
            prompt = f"""
            As a bearish macro researcher, present compelling arguments for macro risks and challenges in Round {round_number}.
            
            ANALYST SCORES TO CONSIDER:
            {self._format_analyst_scores(analyst_scores)}
            
            PREVIOUS BULLISH ARGUMENT TO COUNTER:
            {bullish_argument}
            
            BEARISH RESEARCH FOCUS:
            1. Identify the strongest macro risks based on analyst scores
            2. Highlight negative economic trends and headwinds
            3. Present downside risks in different asset classes
            4. Argue for deflation risks and market pessimism
            5. Emphasize risk-off sentiment and defensive positioning
            6. Counter the bullish arguments with specific risks
            
            RESEARCH REQUIREMENTS:
            - Focus on ETFs with negative scores from analysts
            - Highlight macro trends supporting bearish thesis
            - Present specific economic data and indicators
            - Argue for portfolio positioning in defensive assets
            - Address bullish arguments with specific risk counterarguments
            
            Provide a comprehensive bearish research argument.
            """
            
            response = self.llm(prompt)
            logger.info(f"Bearish researcher completed Round {round_number}")
            
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'perspective': self.perspective,
                'round_number': round_number,
                'research_argument': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Bearish research failed in Round {round_number}: {e}")
            return {
                'agent_name': self.agent_name,
                'specialization': self.specialization,
                'perspective': self.perspective,
                'error': str(e),
                'status': 'failed'
            }
    
    def research_risks(self, analyst_scores: dict, bullish_argument: str, round_number: int) -> str:
        """
        Research and present bearish macro risks.
        
        Args:
            analyst_scores: Dictionary of analyst scores
            bullish_argument: Previous bullish argument to counter
            round_number: Current debate round number
            
        Returns:
            Bearish research argument
        """
        try:
            prompt = f"""
            As a bearish macro researcher, present compelling arguments for macro risks and challenges in Round {round_number}.
            
            ANALYST SCORES TO CONSIDER:
            {self._format_analyst_scores(analyst_scores)}
            
            PREVIOUS BULLISH ARGUMENT TO COUNTER:
            {bullish_argument}
            
            BEARISH RESEARCH FOCUS:
            1. Identify the strongest macro risks based on analyst scores
            2. Highlight negative economic trends and headwinds
            3. Present downside risks in different asset classes
            4. Argue for deflation risks and market pessimism
            5. Emphasize risk-off sentiment and defensive positioning
            6. Counter the bullish arguments with specific risks
            
            RESEARCH REQUIREMENTS:
            - Focus on ETFs with negative scores from analysts
            - Highlight macro trends supporting bearish thesis
            - Present specific economic data and indicators
            - Argue for portfolio positioning in defensive assets
            - Address bullish arguments with specific risk counterarguments
            
            Provide a comprehensive bearish research argument.
            """
            
            response = self.llm(prompt)
            logger.info(f"Bearish researcher completed Round {round_number}")
            return response
            
        except Exception as e:
            logger.error(f"Bearish research failed in Round {round_number}: {e}")
            return f"Bearish research error: {str(e)}"
    
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


def debate(state: dict, rounds: int = None) -> dict:
    """
    Conduct a structured debate between bullish and bearish macro researchers.
    
    Args:
        state: LangGraph state dictionary containing:
            - analyst_scores: Dictionary of analyst scores
            - universe: List of ETFs being analyzed
        rounds: Number of debate rounds (defaults to DEFAULT_CONFIG)
        
    Returns:
        Updated state dictionary with debate results
    """
    if rounds is None:
        rounds = DEFAULT_CONFIG.get('max_debate_rounds', 2)
    
    try:
        # Extract data from state
        analyst_scores = state.get('analyst_scores', {})
        universe = state.get('universe', [])
        
        # Initialize researchers
        bullish_researcher = BullishMacroResearcher("BullishResearcher")
        bearish_researcher = BearishMacroResearcher("BearishResearcher")
        
        # Initialize debate log
        debate_log = []
        debate_log.append(f"=== MACRO ETF DEBATE STARTED ===")
        debate_log.append(f"Universe: {', '.join(universe)}")
        debate_log.append(f"Analysts: {', '.join(analyst_scores.keys())}")
        debate_log.append(f"Rounds: {rounds}")
        debate_log.append("")
        
        logger.info(f"Starting macro debate with {rounds} rounds")
        
        # Conduct debate rounds
        for round_num in range(1, rounds + 1):
            debate_log.append(f"=== ROUND {round_num} ===")
            
            # Bullish argument
            bullish_arg = bullish_researcher.research_opportunities(analyst_scores, round_num)
            debate_log.append(f"BULLISH ARGUMENT:")
            debate_log.append(bullish_arg)
            debate_log.append("")
            
            # Bearish counter-argument
            bearish_arg = bearish_researcher.research_risks(analyst_scores, bullish_arg, round_num)
            debate_log.append(f"BEARISH COUNTER-ARGUMENT:")
            debate_log.append(bearish_arg)
            debate_log.append("")
        
        # Final debate summary
        debate_log.append("=== DEBATE SUMMARY ===")
        debate_log.append("The debate has concluded with comprehensive analysis from both perspectives.")
        debate_log.append("Consider both bullish opportunities and bearish risks in final decisions.")
        
        # Store debate results in state
        state['debate_output'] = debate_log
        state['debate_rounds'] = rounds
        state['debate_participants'] = ['BullishMacroResearcher', 'BearishMacroResearcher']
        state['debate_summary'] = f"Debate completed with {rounds} rounds between bullish and bearish researchers"
        
        logger.info(f"Macro debate completed with {rounds} rounds")
        return state
        
    except Exception as e:
        logger.error(f"Debate failed: {e}")
        # Return state with error information
        state['debate_output'] = [f"Debate error: {str(e)}"]
        state['debate_error'] = str(e)
        return state


def analyze_debate_results(state: dict) -> dict:
    """
    Analyze the results of the debate and provide final recommendations.
    
    Args:
        state: LangGraph state dictionary with debate results
        
    Returns:
        Updated state with analysis of debate results
    """
    try:
        debate_output = state.get('debate_output', [])
        analyst_scores = state.get('analyst_scores', {})
        universe = state.get('universe', [])
        
        if not debate_output:
            logger.warning("No debate output to analyze")
            return state
        
        # Create analysis prompt
        prompt = f"""
        As a macro analysis expert, analyze the following debate results and provide final ETF recommendations.
        
        DEBATE RESULTS:
        {chr(10).join(debate_output)}
        
        ANALYST SCORES:
        {_format_all_analyst_scores(analyst_scores)}
        
        ETF UNIVERSE:
        {', '.join(universe)}
        
        ANALYSIS REQUIREMENTS:
        1. Summarize the key arguments from both bullish and bearish perspectives
        2. Identify the strongest opportunities and risks
        3. Provide final ETF recommendations based on the debate
        4. Highlight consensus areas and areas of disagreement
        5. Suggest portfolio positioning based on the analysis
        
        Provide a comprehensive analysis of the debate results.
        """
        
        # Use a neutral researcher for final analysis
        neutral_researcher = BullishMacroResearcher("NeutralAnalyst")
        analysis = neutral_researcher.llm(prompt)
        
        # Store analysis results
        state['debate_analysis'] = analysis
        state['debate_completed'] = True
        
        logger.info("Debate analysis completed")
        return state
        
    except Exception as e:
        logger.error(f"Debate analysis failed: {e}")
        state['debate_analysis'] = f"Analysis error: {str(e)}"
        return state


def _format_all_analyst_scores(analyst_scores: dict) -> str:
    """Format all analyst scores for analysis."""
    if not analyst_scores:
        return "No analyst scores available"
    
    formatted = []
    for analyst, scores in analyst_scores.items():
        formatted.append(f"\n{analyst.upper()} SCORES:")
        for etf, score in scores.items():
            formatted.append(f"  {etf}: {score:.3f}")
    
    return '\n'.join(formatted)


# Example usage and testing
if __name__ == "__main__":
    print("Debate Researchers Test")
    print("="*40)
    
    try:
        # Test individual researchers
        bullish = BullishMacroResearcher("TestBullish")
        bearish = BearishMacroResearcher("TestBearish")
        
        print("✓ Researchers initialized successfully")
        print(f"  Bullish: {bullish.specialization}")
        print(f"  Bearish: {bearish.specialization}")
        
        # Test with sample data
        sample_analyst_scores = {
            'macro': {'SPY': 0.4, 'QQQ': 0.6, 'TLT': -0.8, 'GLD': 0.7},
            'geo': {'SPY': -0.4, 'QQQ': -0.6, 'TLT': 0.3, 'GLD': 0.7},
            'correlation': {'SPY': 0.1, 'QQQ': 0.0, 'TLT': 0.6, 'GLD': 0.7}
        }
        
        # Test bullish research
        bullish_arg = bullish.research_opportunities(sample_analyst_scores, 1)
        print(f"✓ Bullish research completed")
        print(f"  Response length: {len(bullish_arg)} characters")
        
        # Test bearish research
        bearish_arg = bearish.research_risks(sample_analyst_scores, bullish_arg, 1)
        print(f"✓ Bearish research completed")
        print(f"  Response length: {len(bearish_arg)} characters")
        
        # Test full debate
        sample_state = {
            'analyst_scores': sample_analyst_scores,
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'debate_output': []
        }
        
        result_state = debate(sample_state, rounds=2)
        print(f"✓ Full debate completed")
        print(f"  Rounds: {result_state.get('debate_rounds', 0)}")
        print(f"  Output length: {len(result_state.get('debate_output', []))}")
        
        # Test debate analysis
        final_state = analyze_debate_results(result_state)
        print(f"✓ Debate analysis completed")
        print(f"  Analysis length: {len(final_state.get('debate_analysis', ''))}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Debate researchers test completed!")

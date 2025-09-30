"""
LangGraph Workflow for Macro Trading System
"""

from langgraph.graph import StateGraph, END
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.geopolitical_analyst import GeopoliticalAnalystAgent
from src.agents.correlation_specialist import CorrelationSpecialistAgent
from src.agents.trader_agent import TraderAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.portfolio_optimizer import PortfolioOptimizerAgent
from src.agents.debate_researchers import debate
from src.data_fetchers.macro_fetcher import MacroFetcher
from src.config import MACRO_INDICATORS, DEFAULT_CONFIG
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MacroTradingGraph:
    """LangGraph workflow for macro trading system."""
    
    def __init__(self, debug=False):
        """Initialize the macro trading graph."""
        self.debug = debug
        
        # Initialize components
        self.fetcher = MacroFetcher()
        self.macro_analyst = MacroEconomistAgent("MacroEconomist")
        self.geo_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        self.corr_analyst = CorrelationSpecialistAgent("CorrelationSpecialist")
        self.trader = TraderAgent("Trader")
        self.risk_manager = RiskManagerAgent("RiskManager")
        self.optimizer = PortfolioOptimizerAgent("PortfolioOptimizer")
        
        # Build the graph
        self.graph = StateGraph(state_schema=dict)
        self.add_nodes_and_edges()
        self.compiled = self.graph.compile()
        
        logger.info("Macro trading graph initialized successfully")
    
    def add_nodes_and_edges(self):
        """Add nodes and edges to the LangGraph workflow."""
        
        # Data fetching node
        def fetch_node(state):
            """Fetch all required data for analysis."""
            try:
                logger.info("Starting data fetch...")
                
                # Fetch macro data
                macro_data = self.fetcher.fetch_macro_data(MACRO_INDICATORS)
                state['macro_data'] = macro_data
                
                # Fetch ETF data
                universe = state.get('universe', [])
                etf_data = self.fetcher.fetch_etf_data(universe)
                state['etf_data'] = etf_data
                
                # Fetch news data
                news_data = self.fetcher.fetch_geopolitical_news(
                    query='global macro trends geopolitical events'
                )
                state['news'] = news_data
                
                # Initialize analyst scores and reasoning
                state['analyst_scores'] = {}
                state['agent_reasoning'] = {}
                
                logger.info(f"Data fetch completed for {len(universe)} ETFs")
                return state
                
            except Exception as e:
                logger.error(f"Data fetch failed: {e}")
                # Return state with empty data on error
                state['macro_data'] = {}
                state['etf_data'] = pd.DataFrame()
                state['news'] = []
                state['analyst_scores'] = {}
                state['agent_reasoning'] = {}
                return state
        
        # Analysis nodes
        def macro_analyst_node(state):
            """Run macro economist analysis."""
            try:
                logger.info("Running macro economist analysis...")
                result = self.macro_analyst.analyze(state)
                logger.info("Macro economist analysis completed")
                return state
            except Exception as e:
                logger.error(f"Macro economist analysis failed: {e}")
                return state
        
        def geo_analyst_node(state):
            """Run geopolitical analyst analysis."""
            try:
                logger.info("Running geopolitical analyst analysis...")
                result = self.geo_analyst.analyze(state)
                logger.info("Geopolitical analyst analysis completed")
                return state
            except Exception as e:
                logger.error(f"Geopolitical analyst analysis failed: {e}")
                return state
        
        def corr_analyst_node(state):
            """Run correlation specialist analysis."""
            try:
                logger.info("Running correlation specialist analysis...")
                result = self.corr_analyst.analyze(state)
                logger.info("Correlation specialist analysis completed")
                return state
            except Exception as e:
                logger.error(f"Correlation specialist analysis failed: {e}")
                return state
        
        # Debate node
        def debate_node(state):
            """Run debate between bullish and bearish researchers."""
            try:
                logger.info("Starting debate...")
                state = debate(state, rounds=DEFAULT_CONFIG.get('max_debate_rounds', 2))
                state['agent_reasoning']['debate'] = {
                    'rounds': state.get('debate_output', []),
                    'summary': state.get('debate_summary', 'No debate summary available'),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                logger.info("Debate completed")
                return state
            except Exception as e:
                logger.error(f"Debate failed: {e}")
                return state
        
        # Allocation nodes
        def trader_node(state):
            """Run trader agent for initial allocations."""
            try:
                logger.info("Running trader agent...")
                result = self.trader.propose(state)
                logger.info("Trader agent completed")
                return state
            except Exception as e:
                logger.error(f"Trader agent failed: {e}")
                return state
        
        def risk_node(state):
            """Run risk manager for risk adjustments."""
            try:
                logger.info("Running risk manager...")
                result = self.risk_manager.assess(state)
                logger.info("Risk manager completed")
                return state
            except Exception as e:
                logger.error(f"Risk manager failed: {e}")
                return state
        
        def optimizer_node(state):
            """Run portfolio optimizer for final allocations."""
            try:
                logger.info("Running portfolio optimizer...")
                result = self.optimizer.optimize(state)
                logger.info("Portfolio optimizer completed")
                return state
            except Exception as e:
                logger.error(f"Portfolio optimizer failed: {e}")
                return state
        
        # Add nodes to graph
        self.graph.add_node('fetch', fetch_node)
        self.graph.add_node('macro_analyst', macro_analyst_node)
        self.graph.add_node('geo_analyst', geo_analyst_node)
        self.graph.add_node('corr_analyst', corr_analyst_node)
        self.graph.add_node('debate', debate_node)
        self.graph.add_node('trader', trader_node)
        self.graph.add_node('risk', risk_node)
        self.graph.add_node('optimizer', optimizer_node)
        
        # Add edges to create workflow
        self.graph.set_entry_point('fetch')
        self.graph.add_edge('fetch', 'macro_analyst')
        self.graph.add_edge('macro_analyst', 'geo_analyst')
        self.graph.add_edge('geo_analyst', 'corr_analyst')
        self.graph.add_edge('corr_analyst', 'debate')
        self.graph.add_edge('debate', 'trader')
        self.graph.add_edge('trader', 'risk')
        self.graph.add_edge('risk', 'optimizer')
        self.graph.add_edge('optimizer', END)
        
        logger.info("Graph nodes and edges added successfully")
    
    def propagate(self, universe, date='today'):
        """
        Run the complete macro trading workflow.
        
        Args:
            universe: List of ETF tickers to analyze
            date: Date for analysis (default: 'today')
            
        Returns:
            Dictionary with final ETF allocations
        """
        try:
            logger.info(f"Starting macro trading workflow for {len(universe)} ETFs")
            logger.info(f"Universe: {', '.join(universe)}")
            
            # Create initial state
            initial_state = {
                'universe': universe,
                'date': date,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Run the workflow
            result = self.compiled.invoke(initial_state)
            
            # Extract final allocations and complete state
            final_allocations = result.get('final_allocations', {})
            complete_state = result
            
            logger.info("Macro trading workflow completed successfully")
            logger.info(f"Final allocations: {final_allocations}")
            
            return complete_state
            
        except Exception as e:
            logger.error(f"Macro trading workflow failed: {e}")
            # Return equal allocations on error
            equal_allocation = 100.0 / len(universe)
            return {etf: equal_allocation for etf in universe}


if __name__ == "__main__":
    print("Macro Trading Graph Test")
    print("="*40)
    
    try:
        # Initialize the graph
        graph = MacroTradingGraph(debug=True)
        print("✓ Macro trading graph initialized")
        
        # Test with sample universe
        test_universe = ['SPY', 'QQQ', 'TLT', 'GLD', 'EWJ']
        print(f"✓ Testing with universe: {', '.join(test_universe)}")
        
        # Run the workflow
        final_allocations = graph.propagate(test_universe)
        print(f"✓ Workflow completed")
        print(f"  Final allocations: {final_allocations}")
        
        # Show allocation summary
        if final_allocations:
            print("\nAllocation Summary:")
            print("="*20)
            sorted_allocations = sorted(final_allocations.items(), key=lambda x: x[1], reverse=True)
            for i, (etf, allocation) in enumerate(sorted_allocations, 1):
                print(f"{i}. {etf}: {allocation:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Macro trading graph test completed!")
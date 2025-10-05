"""
LangGraph Workflow for Macro Trading System

Revamped workflow like original ai-hedge-fund but for batch ETFs:
Fetch → MacroAnalyst → GeoAnalyst → Risk → Portfolio
"""

from langgraph.graph import StateGraph, END
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.geopolitical_analyst import GeopoliticalAnalystAgent
from src.agents.risk_manager import RiskManager
from src.agents.portfolio_agent import PortfolioAgent
from src.data_fetchers.macro_fetcher import MacroFetcher
from src.config import MACRO_INDICATORS, DEFAULT_CONFIG
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MacroTradingGraph:
    """LangGraph workflow for macro trading system - revamped for batch ETFs."""
    
    def __init__(self, debug=False):
        """Initialize the macro trading graph."""
        self.debug = debug
        
        # Initialize components
        self.fetcher = MacroFetcher()
        
        # Build the graph
        self.graph = StateGraph(state_schema=dict)
        self.add_nodes_and_edges()
        self.compiled = self.graph.compile()
        
        logger.info("Macro trading graph initialized successfully")
    
    def add_nodes_and_edges(self):
        """Add nodes and edges to the LangGraph workflow - simplified for batch ETFs."""
        
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
                
                # Fetch comprehensive news data from multiple sources
                news_data = self.fetcher.fetch_comprehensive_geopolitical_news(days_back=30)
                state['news'] = news_data
                
                # Initialize agent reasoning
                state['agent_reasoning'] = {}
                
                logger.info(f"Data fetch completed for {len(universe)} ETFs")
                return state
                
            except Exception as e:
                logger.error(f"Data fetch failed: {e}")
                # Return state with empty data on error
                state['macro_data'] = {}
                state['etf_data'] = pd.DataFrame()
                state['news'] = []
                state['agent_reasoning'] = {}
                return state
        
        # Add nodes to graph - simplified workflow
        self.graph.add_node('fetch', fetch_node)
        self.graph.add_node('macro_analyst', MacroEconomistAgent().analyze)
        self.graph.add_node('geo_analyst', GeopoliticalAnalystAgent().analyze)
        self.graph.add_node('risk', RiskManager().assess)
        self.graph.add_node('portfolio', PortfolioAgent().manage)
        
        # Add edges to create workflow
        self.graph.set_entry_point('fetch')
        self.graph.add_edge('fetch', 'macro_analyst')
        self.graph.add_edge('macro_analyst', 'geo_analyst')
        self.graph.add_edge('geo_analyst', 'risk')
        self.graph.add_edge('risk', 'portfolio')
        self.graph.add_edge('portfolio', END)
        
        logger.info("Graph nodes and edges added successfully")
    
    def propagate(self, universe, date='today'):
        """
        Run the complete macro trading workflow for batch ETFs.
        
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
            
            # Extract final allocations
            final_allocations = result.get('final_allocations', {})
            
            logger.info("Macro trading workflow completed successfully")
            logger.info(f"Final allocations: {final_allocations}")
            
            return final_allocations
            
        except Exception as e:
            logger.error(f"Macro trading workflow failed: {e}")
            # Return equal allocations on error
            equal_allocation = 1.0 / len(universe)
            return {etf: {'action': 'hold', 'allocation': equal_allocation, 'reason': 'Workflow failed'} for etf in universe}
    
    def propagate_with_details(self, universe, date='today'):
        """
        Run the complete macro trading workflow and return detailed results.
        
        Args:
            universe: List of ETF tickers to analyze
            date: Date for analysis (default: 'today')
            
        Returns:
            Dictionary with complete analysis results including all agent reasoning
        """
        try:
            logger.info(f"Starting detailed macro trading workflow for {len(universe)} ETFs")
            logger.info(f"Universe: {', '.join(universe)}")
            
            # Create initial state
            initial_state = {
                'universe': universe,
                'date': date,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Run the workflow
            result = self.compiled.invoke(initial_state)
            
            logger.info("Detailed macro trading workflow completed successfully")
            logger.info(f"Result keys: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Detailed macro trading workflow failed: {e}")
            # Return error state
            return {
                'final_allocations': {etf: {'action': 'hold', 'allocation': 0.0, 'reason': 'Workflow failed'} for etf in universe},
                'agent_reasoning': {},
                'macro_data': {},
                'etf_data': pd.DataFrame(),
                'news': [],
                'error': str(e)
            }


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
            print("\nPortfolio Summary:")
            print("="*20)
            for etf, data in final_allocations.items():
                if isinstance(data, dict):
                    action = data.get('action', 'unknown')
                    allocation = data.get('allocation', 0.0)
                    reason = data.get('reason', 'No reason')[:50]
                    print(f"{etf}: {action.upper()} {allocation:.1%} - {reason}...")
                else:
                    print(f"{etf}: {data:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*40)
    print("Macro trading graph test completed!")
"""
Global Macro ETF Trading System

Revamped main entry point for the global macro ETF trading system.
Runs the complete LangGraph workflow: Fetch → MacroAnalyst → GeoAnalyst → Risk → Portfolio
"""

import argparse
import logging
from src.graph.macro_trading_graph import MacroTradingGraph
from src.config import ETF_UNIVERSE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the revamped global macro ETF trading system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the global macro ETF trading system")
    parser.add_argument('--universe', default=ETF_UNIVERSE, nargs='+', help='List of ETFs for batch analysis')
    parser.add_argument('--date', default='today', help='Date for analysis (default: today)')
    args = parser.parse_args()
    
    try:
        # Initialize the macro trading graph
        logger.info("Initializing global macro ETF trading system...")
        graph = MacroTradingGraph(debug=True)
        logger.info("✓ Macro trading graph initialized successfully")
        
        # Run the complete workflow
        logger.info("Starting macro trading workflow...")
        allocations = graph.propagate(args.universe, args.date)
        
        # Display results
        print(f"Final Allocations: {allocations}")
        
        # Show portfolio summary
        if allocations:
            print("\nPortfolio Summary:")
            print("="*20)
            for etf, data in allocations.items():
                if isinstance(data, dict):
                    action = data.get('action', 'unknown')
                    allocation = data.get('allocation', 0.0)
                    reason = data.get('reason', 'No reason')[:50]
                    print(f"{etf}: {action.upper()} {allocation:.1%} - {reason}...")
                else:
                    print(f"{etf}: {data:.1%}")
        
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

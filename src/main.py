"""
Global Macro ETF Trading System

This is the main entry point for the global macro ETF trading system.
It runs the complete LangGraph workflow from data fetching to portfolio optimization.
"""

import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

from src.graph.macro_trading_graph import MacroTradingGraph
from src.config import ETF_UNIVERSE

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the global macro ETF trading system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the global macro ETF trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                                    # Use default ETF universe
  python src/main.py --universe SPY QQQ TLT GLD        # Custom ETF universe
  python src/main.py --universe SPY QQQ --date 2024-01-01  # Custom date
        """
    )
    
    parser.add_argument(
        '--universe', 
        default=ETF_UNIVERSE, 
        nargs='+', 
        help=f'List of ETFs for batch analysis (default: {ETF_UNIVERSE[:5]}...)'
    )
    
    parser.add_argument(
        '--date', 
        default='today',
        help='Date for analysis (default: today)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Initialize the macro trading graph
        logger.info("Initializing global macro ETF trading system...")
        graph = MacroTradingGraph(debug=args.debug)
        logger.info("✓ Macro trading graph initialized successfully")
        
        # Display configuration
        logger.info(f"ETF Universe: {', '.join(args.universe)}")
        logger.info(f"Analysis Date: {args.date}")
        logger.info(f"Number of ETFs: {len(args.universe)}")
        
        # Run the complete workflow
        logger.info("Starting macro trading workflow...")
        start_time = datetime.now()
        
        allocations = graph.propagate(args.universe, args.date)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print("\n" + "="*60)
        print("GLOBAL MACRO ETF TRADING SYSTEM - RESULTS")
        print("="*60)
        print(f"Analysis completed in {duration:.1f} seconds")
        print(f"Date: {args.date}")
        print(f"Universe: {', '.join(args.universe)}")
        print()
        
        if allocations:
            print("Final Macro Allocations (buy percentages):")
            print("-" * 45)
            
            # Sort allocations by percentage (descending)
            sorted_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
            
            for i, (etf, allocation) in enumerate(sorted_allocations, 1):
                print(f"{i:2d}. {etf:6s}: {allocation:6.1f}%")
            
            # Verify allocations sum to 100%
            total_allocation = sum(allocations.values())
            print("-" * 45)
            print(f"Total: {total_allocation:.1f}%")
            
            if abs(total_allocation - 100.0) < 0.1:
                print("✓ Allocations properly normalized")
            else:
                print("⚠ Warning: Allocations may not be properly normalized")
            
            # Show top recommendations
            print("\nTop Recommendations:")
            top_3 = sorted_allocations[:3]
            for i, (etf, allocation) in enumerate(top_3, 1):
                print(f"  {i}. {etf}: {allocation:.1f}%")
            
        else:
            print("❌ No allocations generated")
            print("This may indicate an error in the workflow")
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Analysis interrupted by user")
        logger.info("Analysis interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

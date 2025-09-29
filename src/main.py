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


def display_comprehensive_reasoning(complete_state, allocations):
    """Display comprehensive reasoning from all agents and allocation rationale."""
    
    agent_reasoning = complete_state.get('agent_reasoning', {})
    
    # Display final allocations
    if allocations:
        print("📊 FINAL MACRO ALLOCATIONS (Buy Percentages):")
        print("-" * 50)
        
        # Sort allocations by percentage (descending)
        sorted_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        for i, (etf, allocation) in enumerate(sorted_allocations, 1):
            print(f"{i:2d}. {etf:6s}: {allocation:6.1f}%")
        
        # Verify allocations sum to 100%
        total_allocation = sum(allocations.values())
        print("-" * 50)
        print(f"Total: {total_allocation:.1f}%")
        
        if abs(total_allocation - 100.0) < 0.1:
            print("✓ Allocations properly normalized")
        else:
            print("⚠ Warning: Allocations may not be properly normalized")
        
        # Show top recommendations
        print("\n🎯 TOP RECOMMENDATIONS:")
        top_3 = sorted_allocations[:3]
        for i, (etf, allocation) in enumerate(top_3, 1):
            print(f"  {i}. {etf}: {allocation:.1f}%")
    else:
        print("❌ No allocations generated")
        print("This may indicate an error in the workflow")
        return
    
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE ANALYSIS REASONING")
    print("="*60)
    
    # Display macro economist reasoning
    if 'macro_economist' in agent_reasoning:
        macro_data = agent_reasoning['macro_economist']
        print("\n🏛️  MACRO ECONOMIST ANALYSIS:")
        print("-" * 40)
        print(f"Key Factors: {', '.join(macro_data.get('key_factors', ['Not specified']))}")
        print(f"Reasoning: {macro_data.get('reasoning', 'No detailed reasoning provided')}")
        if macro_data.get('scores'):
            print("ETF Scores:")
            for etf, score in macro_data['scores'].items():
                print(f"  {etf}: {score:.2f}")
    
    # Display geopolitical analyst reasoning
    if 'geopolitical_analyst' in agent_reasoning:
        geo_data = agent_reasoning['geopolitical_analyst']
        print("\n🌍 GEOPOLITICAL ANALYST ANALYSIS:")
        print("-" * 40)
        print(f"Key Factors: {', '.join(geo_data.get('key_factors', ['Not specified']))}")
        print(f"Reasoning: {geo_data.get('reasoning', 'No detailed reasoning provided')}")
        if geo_data.get('scores'):
            print("ETF Scores:")
            for etf, score in geo_data['scores'].items():
                print(f"  {etf}: {score:.2f}")
    
    # Display correlation specialist reasoning
    if 'correlation_specialist' in agent_reasoning:
        corr_data = agent_reasoning['correlation_specialist']
        print("\n📈 CORRELATION SPECIALIST ANALYSIS:")
        print("-" * 40)
        print(f"Key Factors: {', '.join(corr_data.get('key_factors', ['Not specified']))}")
        print(f"Reasoning: {corr_data.get('reasoning', 'No detailed reasoning provided')}")
        if corr_data.get('scores'):
            print("ETF Scores:")
            for etf, score in corr_data['scores'].items():
                print(f"  {etf}: {score:.2f}")
    
    # Display debate results
    if 'debate' in agent_reasoning:
        debate_data = agent_reasoning['debate']
        print("\n⚔️  BULLISH vs BEARISH DEBATE:")
        print("-" * 40)
        print(f"Summary: {debate_data.get('summary', 'No debate summary available')}")
        if debate_data.get('rounds'):
            print(f"Debate Rounds: {len(debate_data['rounds'])}")
            for i, round_data in enumerate(debate_data['rounds'][:2], 1):  # Show first 2 rounds
                print(f"  Round {i}: {round_data[:100]}..." if len(round_data) > 100 else f"  Round {i}: {round_data}")
    
    # Display trader reasoning
    if 'trader' in agent_reasoning:
        trader_data = agent_reasoning['trader']
        print("\n💼 TRADER AGENT REASONING:")
        print("-" * 40)
        print(f"Key Factors: {', '.join(trader_data.get('key_factors', ['Not specified']))}")
        print(f"Reasoning: {trader_data.get('reasoning', 'No detailed reasoning provided')}")
        if trader_data.get('proposed_allocations'):
            print("Proposed Allocations:")
            for etf, allocation in trader_data['proposed_allocations'].items():
                print(f"  {etf}: {allocation:.1f}%")
    
    # Display risk manager reasoning
    if 'risk_manager' in agent_reasoning:
        risk_data = agent_reasoning['risk_manager']
        print("\n🛡️  RISK MANAGER ASSESSMENT:")
        print("-" * 40)
        print(f"Risk Factors: {', '.join(risk_data.get('risk_factors', ['Not specified']))}")
        print(f"Reasoning: {risk_data.get('reasoning', 'No detailed reasoning provided')}")
        if risk_data.get('adjustments'):
            print("Risk Adjustments:")
            for etf, adjustment in risk_data['adjustments'].items():
                print(f"  {etf}: {adjustment}")
        if risk_data.get('risk_adjusted_allocations'):
            print("Risk-Adjusted Allocations:")
            for etf, allocation in risk_data['risk_adjusted_allocations'].items():
                print(f"  {etf}: {allocation:.1f}%")
    
    # Display portfolio optimizer reasoning
    if 'portfolio_optimizer' in agent_reasoning:
        opt_data = agent_reasoning['portfolio_optimizer']
        print("\n⚖️  PORTFOLIO OPTIMIZER ANALYSIS:")
        print("-" * 40)
        print(f"Optimization Method: {opt_data.get('optimization_method', 'Not specified')}")
        print(f"Reasoning: {opt_data.get('reasoning', 'No detailed reasoning provided')}")
        if opt_data.get('constraints'):
            print("Constraints Applied:")
            for constraint, value in opt_data['constraints'].items():
                print(f"  {constraint}: {value}")
        if opt_data.get('performance_metrics'):
            print("Performance Metrics:")
            for metric, value in opt_data['performance_metrics'].items():
                print(f"  {metric}: {value}")
    
    # Display allocation rationale summary
    print("\n" + "="*60)
    print("🎯 ALLOCATION RATIONALE SUMMARY")
    print("="*60)
    
    if allocations:
        print("\nWhy these allocations were recommended:")
        print("-" * 50)
        
        # Analyze the reasoning to provide a summary
        rationale_summary = generate_allocation_rationale(complete_state, allocations)
        print(rationale_summary)
        
        print("\n💡 KEY INSIGHTS:")
        print("-" * 20)
        insights = generate_key_insights(complete_state, allocations)
        for insight in insights:
            print(f"• {insight}")


def generate_allocation_rationale(complete_state, allocations):
    """Generate a comprehensive rationale for the allocations."""
    
    agent_reasoning = complete_state.get('agent_reasoning', {})
    rationale_parts = []
    
    # Get macro factors
    if 'macro_economist' in agent_reasoning:
        macro_factors = agent_reasoning['macro_economist'].get('key_factors', [])
        if macro_factors:
            rationale_parts.append(f"Macro factors: {', '.join(macro_factors)}")
    
    # Get geopolitical factors
    if 'geopolitical_analyst' in agent_reasoning:
        geo_factors = agent_reasoning['geopolitical_analyst'].get('key_factors', [])
        if geo_factors:
            rationale_parts.append(f"Geopolitical factors: {', '.join(geo_factors)}")
    
    # Get risk factors
    if 'risk_manager' in agent_reasoning:
        risk_factors = agent_reasoning['risk_manager'].get('risk_factors', [])
        if risk_factors:
            rationale_parts.append(f"Risk considerations: {', '.join(risk_factors)}")
    
    # Generate ETF-specific rationale
    for etf, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        if allocation > 0:
            rationale_parts.append(f"{etf} ({allocation:.1f}%): Recommended based on macro trends and risk assessment")
    
    return "\n".join(rationale_parts) if rationale_parts else "No detailed rationale available"


def generate_key_insights(complete_state, allocations):
    """Generate key insights from the analysis."""
    
    insights = []
    
    # Top allocation insight
    if allocations:
        top_etf = max(allocations.items(), key=lambda x: x[1])
        insights.append(f"Highest allocation: {top_etf[0]} ({top_etf[1]:.1f}%) - Primary focus based on analysis")
    
    # Diversification insight
    if len(allocations) > 1:
        insights.append(f"Portfolio includes {len(allocations)} ETFs for diversification")
    
    # Risk insight
    agent_reasoning = complete_state.get('agent_reasoning', {})
    if 'risk_manager' in agent_reasoning:
        risk_factors = agent_reasoning['risk_manager'].get('risk_factors', [])
        if risk_factors:
            insights.append(f"Risk management applied: {', '.join(risk_factors[:2])}")
    
    # Macro insight
    if 'macro_economist' in agent_reasoning:
        macro_factors = agent_reasoning['macro_economist'].get('key_factors', [])
        if macro_factors:
            insights.append(f"Key macro drivers: {', '.join(macro_factors[:2])}")
    
    return insights if insights else ["Analysis completed successfully"]


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
        
        complete_state = graph.propagate(args.universe, args.date)
        allocations = complete_state.get('final_allocations', {})
        
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
        
        # Display comprehensive reasoning
        display_comprehensive_reasoning(complete_state, allocations)
        
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

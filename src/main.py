"""
Global Macro ETF Trading System

Revamped main entry point for the global macro ETF trading system.
Runs the complete LangGraph workflow: Fetch → MacroAnalyst → GeoAnalyst → Risk → Portfolio
"""

import argparse
import logging
import json
import os
from datetime import datetime
from src.graph.macro_trading_graph import MacroTradingGraph
from src.config import ETF_UNIVERSE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_final_report(result, universe, date):
    """Generate a comprehensive final report with all agent reasoning and final allocations."""
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"macro_analysis_report_{timestamp}.txt"
    report_path = os.path.join(reports_dir, report_filename)
    
    # Create report content
    report_content = []
    
    def add_line(text=""):
        report_content.append(text)
        print(text)
    
    add_line("\n" + "="*80)
    add_line("🤖 AI GLOBAL MACRO FUND - COMPREHENSIVE ANALYSIS REPORT")
    add_line("="*80)
    add_line(f"📅 Analysis Date: {date}")
    add_line(f"📊 Universe: {', '.join(universe)}")
    add_line(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add_line("="*80)
    
    # Extract data from result
    final_allocations = result.get('final_allocations', {})
    agent_reasoning = result.get('agent_reasoning', {})
    macro_data = result.get('macro_data', {})
    etf_data = result.get('etf_data', {})
    news = result.get('news', [])
    
    # 1. MACRO ECONOMIC OVERVIEW
    add_line("\n📈 MACRO ECONOMIC OVERVIEW")
    add_line("-" * 50)
    if macro_data:
        add_line("Key Economic Indicators:")
        key_indicators = ['CPIAUCSL', 'UNRATE', 'GDPC1', 'FEDFUNDS', 'VIXCLS', 'DGS10']
        for indicator in key_indicators:
            if indicator in macro_data:
                value = macro_data[indicator]
                add_line(f"  • {indicator}: {value}")
    else:
        add_line("  • Macro data not available")
    
    # 2. GEOPOLITICAL NEWS SUMMARY
    add_line("\n🌍 GEOPOLITICAL NEWS SUMMARY")
    add_line("-" * 50)
    if news:
        add_line(f"Total Articles Analyzed: {len(news)}")
        add_line("Key News Categories:")
        categories = {}
        for article in news[:10]:  # Show top 10 articles
            category = article.get('category', 'General')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        for cat, count in categories.items():
            add_line(f"  • {cat}: {count} articles")
    else:
        add_line("  • No geopolitical news data available")
    
    # 3. MACRO ECONOMIST ANALYSIS
    add_line("\n🏛️ MACRO ECONOMIST ANALYSIS")
    add_line("-" * 50)
    macro_scores = result.get('macro_scores', {})
    if macro_scores:
        add_line("ETF Scores from Macro Economist:")
        for etf, data in macro_scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0.0)
                confidence = data.get('confidence', 0.0)
                reason = data.get('reason', 'No reasoning provided')
                add_line(f"  • {etf}: Score {score:.3f} (Confidence: {confidence:.1%})")
                add_line(f"    Reasoning: {reason[:100]}...")
    else:
        add_line("  • Macro economist analysis not available")
    
    # 4. GEOPOLITICAL ANALYST ANALYSIS
    add_line("\n🗺️ GEOPOLITICAL ANALYST ANALYSIS")
    add_line("-" * 50)
    geo_scores = result.get('geo_scores', {})
    if geo_scores:
        add_line("ETF Scores from Geopolitical Analyst:")
        for etf, data in geo_scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0.0)
                confidence = data.get('confidence', 0.0)
                reason = data.get('reason', 'No reasoning provided')
                add_line(f"  • {etf}: Score {score:.3f} (Confidence: {confidence:.1%})")
                add_line(f"    Reasoning: {reason[:100]}...")
    else:
        add_line("  • Geopolitical analyst analysis not available")
    
    # 5. RISK MANAGER ASSESSMENT
    add_line("\n⚠️ RISK MANAGER ASSESSMENT")
    add_line("-" * 50)
    risk_assessments = result.get('risk_assessments', {})
    if risk_assessments:
        add_line("Risk Assessments:")
        for etf, data in risk_assessments.items():
            if isinstance(data, dict):
                risk_level = data.get('risk_level', 'medium')
                adjusted_score = data.get('adjusted_score', 0.0)
                reason = data.get('reason', 'No reasoning provided')
                add_line(f"  • {etf}: {risk_level.upper()} Risk (Adjusted Score: {adjusted_score:.3f})")
                add_line(f"    Reasoning: {reason[:100]}...")
    else:
        add_line("  • Risk assessments not available")
    
    # 6. FINAL PORTFOLIO ALLOCATIONS
    add_line("\n💼 FINAL PORTFOLIO ALLOCATIONS")
    add_line("-" * 50)
    if final_allocations:
        total_allocation = sum([data.get('allocation', 0.0) for data in final_allocations.values() if isinstance(data, dict)])
        add_line(f"Total Portfolio Allocation: {total_allocation:.1%}")
        add_line("\nDetailed Allocations:")
        
        for etf, data in final_allocations.items():
            if isinstance(data, dict):
                action = data.get('action', 'unknown').upper()
                allocation = data.get('allocation', 0.0)
                reason = data.get('reason', 'No reasoning provided')
                add_line(f"\n  📊 {etf} ({action} {allocation:.1%})")
                add_line(f"     Reasoning: {reason}")
            else:
                add_line(f"\n  📊 {etf}: {data:.1%}")
    else:
        add_line("  • Final allocations not available")
    
    # 7. PORTFOLIO SUMMARY
    add_line("\n📋 PORTFOLIO SUMMARY")
    add_line("-" * 50)
    if final_allocations:
        buy_positions = [etf for etf, data in final_allocations.items() 
                        if isinstance(data, dict) and data.get('action') == 'buy' and data.get('allocation', 0) > 0]
        hold_positions = [etf for etf, data in final_allocations.items() 
                         if isinstance(data, dict) and data.get('action') == 'hold']
        
        add_line(f"Active Positions: {len(buy_positions)}")
        if buy_positions:
            add_line(f"  • Buy: {', '.join(buy_positions)}")
        add_line(f"Hold Positions: {len(hold_positions)}")
        if hold_positions:
            add_line(f"  • Hold: {', '.join(hold_positions)}")
    
    # 8. AGENT REASONING SUMMARY
    add_line("\n🤖 AGENT REASONING SUMMARY")
    add_line("-" * 50)
    if agent_reasoning:
        for agent_name, reasoning in agent_reasoning.items():
            add_line(f"\n{agent_name.replace('_', ' ').title()}:")
            if isinstance(reasoning, dict):
                for key, value in reasoning.items():
                    if key != 'timestamp':
                        add_line(f"  • {key}: {str(value)[:100]}...")
    
    # 9. FINAL RECOMMENDATIONS
    add_line("\n🎯 FINAL RECOMMENDATIONS")
    add_line("-" * 50)
    if final_allocations:
        add_line("Based on comprehensive analysis by all agents:")
        for etf, data in final_allocations.items():
            if isinstance(data, dict):
                action = data.get('action', 'unknown').upper()
                allocation = data.get('allocation', 0.0)
                if action == 'BUY' and allocation > 0:
                    add_line(f"  ✅ {etf}: {action} {allocation:.1%} - Strong conviction based on macro and geopolitical factors")
                elif action == 'HOLD':
                    add_line(f"  ⏸️ {etf}: {action} - Cautious approach due to risk factors")
                else:
                    add_line(f"  ❌ {etf}: {action} - Avoid due to negative signals")
    
    add_line("\n" + "="*80)
    add_line("✅ ANALYSIS COMPLETED SUCCESSFULLY")
    add_line("="*80)
    
    # Save report to file
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        add_line(f"\n📁 Report saved to: {report_path}")
        logger.info(f"Report saved to: {report_path}")
    except Exception as e:
        add_line(f"\n❌ Failed to save report: {e}")
        logger.error(f"Failed to save report: {e}")


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
        result = graph.propagate_with_details(args.universe, args.date)
        
        # Generate comprehensive report
        generate_final_report(result, args.universe, args.date)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

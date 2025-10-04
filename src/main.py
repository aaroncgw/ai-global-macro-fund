"""
Global Macro ETF Trading System

This is the main entry point for the global macro ETF trading system.
It runs the complete LangGraph workflow from data fetching to portfolio optimization.
"""

import argparse
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

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
    """Display comprehensive reasoning from all agents with enhanced analysis."""
    agent_reasoning = complete_state.get('agent_reasoning', {})
    macro_data = complete_state.get('macro_data', {})
    news_data = complete_state.get('news', [])
    debate_output = complete_state.get('debate_output', [])
    
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
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MACRO ANALYSIS REPORT")
    print("="*80)
    
    # 1. MACRO ECONOMIC ANALYSIS
    print("\n🏛️  MACRO ECONOMIC ANALYSIS")
    print("="*50)
    
    # Show ETF data lookback information
    etf_data = complete_state.get('etf_data', pd.DataFrame())
    if not etf_data.empty:
        # Calculate actual lookback period
        if hasattr(etf_data.index, 'min') and hasattr(etf_data.index, 'max'):
            start_date = etf_data.index.min()
            end_date = etf_data.index.max()
            years_of_data = (end_date - start_date).days / 365.25
            print(f"📊 ETF Historical Data:")
            print(f"  • Lookback Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  • Years of Data: {years_of_data:.1f} years")
            print(f"  • ETFs Analyzed: {len(etf_data.columns.get_level_values(0).unique()) if isinstance(etf_data.columns, pd.MultiIndex) else len(etf_data.columns)}")
            print(f"  • Data Points: {len(etf_data)} trading days")
        else:
            print(f"📊 ETF Historical Data: {len(etf_data)} data points available")
    else:
        print("📊 ETF Historical Data: No data available")
    
    if macro_data:
        print("\n📈 Key Economic Indicators:")
        for indicator, data in macro_data.items():
            if 'error' not in data and data.get('latest_value') is not None:
                latest_value = data.get('latest_value', 'N/A')
                periods = data.get('periods', 0)
                print(f"  • {indicator}: {latest_value} ({periods} periods of data)")
            else:
                print(f"  • {indicator}: Error - {data.get('error', 'No data')}")
    else:
        print("  No macro economic data available")
    
    # Display macro economist reasoning
    if 'macro_economist' in agent_reasoning:
        macro_agent = agent_reasoning['macro_economist']
        print(f"\n🎯 Macro Economist Assessment:")
        print(f"  Reasoning: {macro_agent.get('reasoning', 'No detailed reasoning provided')}")
        if macro_agent.get('key_factors'):
            print("  Key Macro Factors:")
            for factor in macro_agent['key_factors']:
                print(f"    • {factor}")
        if macro_agent.get('scores'):
            print("  ETF Scores with Macro Analysis:")
            for etf, score in macro_agent['scores'].items():
                sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                print(f"    {etf}: {score:.2f} {sentiment}")
                # Add specific macro reasoning for each ETF
                if etf == 'SPY':
                    print(f"      → S&P 500: Affected by overall economic growth, inflation, and Fed policy")
                elif etf == 'QQQ':
                    print(f"      → NASDAQ: Tech-heavy, sensitive to interest rates and growth expectations")
                elif etf == 'TLT':
                    print(f"      → Treasury Bonds: Directly impacted by Fed rates and inflation expectations")
                elif etf == 'GLD':
                    print(f"      → Gold: Safe haven asset, inversely correlated with real rates")
                elif etf == 'EWJ':
                    print(f"      → Japan ETF: Affected by USD/JPY, BoJ policy, and global growth")
                elif etf == 'EWG':
                    print(f"      → Germany ETF: European growth, ECB policy, and trade relations")
                elif etf == 'FXI':
                    print(f"      → China ETF: Chinese economic data, trade tensions, and regulatory changes")
    
    # 2. GEOPOLITICAL ANALYSIS
    print("\n🌍 GEOPOLITICAL ANALYSIS")
    print("="*50)
    if news_data:
        print("📰 Key News Events:")
        for i, article in enumerate(news_data[:3], 1):  # Show top 3 articles
            title = article.get('title', 'Unknown Title')[:80]
            sentiment = article.get('sentiment', 'neutral')
            print(f"  {i}. {title}...")
            print(f"     Sentiment: {sentiment}")
    else:
        print("  No geopolitical news data available")
    
    # Display geopolitical analyst reasoning
    if 'geopolitical_analyst' in agent_reasoning:
        geo_agent = agent_reasoning['geopolitical_analyst']
        print(f"\n🎯 Geopolitical Analyst Assessment:")
        print(f"  Reasoning: {geo_agent.get('reasoning', 'No detailed reasoning provided')}")
        if geo_agent.get('key_factors'):
            print("  Key Geopolitical Factors:")
            for factor in geo_agent['key_factors']:
                print(f"    • {factor}")
        if geo_agent.get('scores'):
            print("  ETF Scores with Geopolitical Analysis:")
            for etf, score in geo_agent['scores'].items():
                sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                print(f"    {etf}: {score:.2f} {sentiment}")
                # Connect news articles to specific ETF impacts
                if etf == 'SPY':
                    print(f"      → S&P 500: Affected by US-China trade tensions, domestic political stability")
                elif etf == 'QQQ':
                    print(f"      → NASDAQ: Tech regulation, China trade, semiconductor restrictions")
                elif etf == 'TLT':
                    print(f"      → Treasury Bonds: Safe haven during geopolitical tensions, flight to quality")
                elif etf == 'GLD':
                    print(f"      → Gold: Traditional safe haven, benefits from global uncertainty")
                elif etf == 'EWJ':
                    print(f"      → Japan ETF: US-Japan relations, China tensions, regional stability")
                elif etf == 'EWG':
                    print(f"      → Germany ETF: EU-China relations, energy security, trade wars")
                elif etf == 'FXI':
                    print(f"      → China ETF: US-China tensions, regulatory crackdowns, trade policies")
    
    # 3. CORRELATION & DIVERSIFICATION ANALYSIS
    print("\n📈 CORRELATION & DIVERSIFICATION ANALYSIS")
    print("="*50)
    if 'correlation_specialist' in agent_reasoning:
        corr_agent = agent_reasoning['correlation_specialist']
        print(f"🎯 Correlation Specialist Assessment:")
        print(f"  Reasoning: {corr_agent.get('reasoning', 'No detailed reasoning provided')}")
        if corr_agent.get('scores'):
            print("  Diversification Scores:")
            for etf, score in corr_agent['scores'].items():
                diversification = "🟢 High Diversification" if score > 0.3 else "🔴 Low Diversification" if score < -0.3 else "⚪ Moderate"
                print(f"    {etf}: {score:.2f} {diversification}")
    
    # 4. BULLISH vs BEARISH DEBATE ANALYSIS
    print("\n⚔️  BULLISH vs BEARISH DEBATE ANALYSIS")
    print("="*50)
    if debate_output:
        print("🗣️  Full Debate Content:")
        for i, line in enumerate(debate_output, 1):
            if line.strip():
                print(f"  {i:2d}. {line}")
    else:
        print("  No debate data available")
    
    # 5. RISK ASSESSMENT
    print("\n🛡️  RISK ASSESSMENT")
    print("="*50)
    if 'risk_manager' in agent_reasoning:
        risk_agent = agent_reasoning['risk_manager']
        print(f"🎯 Risk Manager Assessment:")
        print(f"  Reasoning: {risk_agent.get('reasoning', 'No detailed reasoning provided')}")
        if risk_agent.get('risk_factors'):
            print("  Key Risk Factors:")
            for factor in risk_agent['risk_factors']:
                print(f"    • {factor}")
        if risk_agent.get('adjustments'):
            print("  Risk Adjustments Applied:")
            for etf, adjustment in risk_agent['adjustments'].items():
                print(f"    {etf}: {adjustment}")
    
    # 6. PORTFOLIO OPTIMIZATION
    print("\n⚖️  PORTFOLIO OPTIMIZATION")
    print("="*50)
    if 'portfolio_optimizer' in agent_reasoning:
        opt_agent = agent_reasoning['portfolio_optimizer']
        print(f"🎯 Portfolio Optimizer Assessment:")
        print(f"  Method: {opt_agent.get('optimization_method', 'Not specified')}")
        print(f"  Reasoning: {opt_agent.get('reasoning', 'No detailed reasoning provided')}")
        if opt_agent.get('performance_metrics'):
            print("  Performance Metrics:")
            for metric, value in opt_agent['performance_metrics'].items():
                print(f"    {metric}: {value}")
    
    # 7. COMPREHENSIVE ETF ANALYSIS
    print("\n🎯 COMPREHENSIVE ETF ANALYSIS")
    print("="*50)
    
    if allocations:
        # Get all scores for comprehensive analysis
        all_scores = {}
        for agent_name, agent_data in agent_reasoning.items():
            if agent_data.get('scores'):
                all_scores[agent_name] = agent_data['scores']
        
        print("📊 ETF-by-ETF Analysis:")
        for etf in sorted(allocations.keys(), key=lambda x: allocations[x], reverse=True):
            allocation = allocations[etf]
            print(f"\n  📈 {etf} ({allocation:.1f}% allocation):")
            
            # Collect all scores for this ETF
            etf_scores = {}
            for agent_name, scores in all_scores.items():
                if etf in scores:
                    etf_scores[agent_name] = scores[etf]
            
            # Display scores
            if etf_scores:
                print("    Scores by Analyst:")
                for agent_name, score in etf_scores.items():
                    agent_display = agent_name.replace('_', ' ').title()
                    sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                    print(f"      {agent_display}: {score:.2f} {sentiment}")
            
            # Provide rationale for allocation
            if allocation > 20:
                print(f"    💡 HIGH ALLOCATION: {etf} receives {allocation:.1f}% due to strong positive signals across multiple analysts")
            elif allocation > 10:
                print(f"    💡 MODERATE ALLOCATION: {etf} receives {allocation:.1f}% with mixed but generally positive signals")
            elif allocation > 5:
                print(f"    💡 LOW ALLOCATION: {etf} receives {allocation:.1f}% with cautious positioning")
            else:
                print(f"    💡 MINIMAL ALLOCATION: {etf} receives {allocation:.1f}% due to negative signals or high risk")
    
    # 8. FINAL RECOMMENDATIONS
    print("\n🎯 FINAL RECOMMENDATIONS & RATIONALE")
    print("="*50)
    
    if allocations:
        print("📋 Portfolio Allocation Rationale:")
        print()
        
        # Sort ETFs by allocation
        sorted_etfs = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        for i, (etf, allocation) in enumerate(sorted_etfs, 1):
            print(f"{i}. {etf} ({allocation:.1f}%):")
            
            # Get reasoning for this ETF
            etf_reasoning = []
            for agent_name, agent_data in agent_reasoning.items():
                if agent_data.get('scores') and etf in agent_data['scores']:
                    score = agent_data['scores'][etf]
                    agent_display = agent_name.replace('_', ' ').title()
                    if score > 0.1:
                        etf_reasoning.append(f"Strong positive signal from {agent_display}")
                    elif score < -0.1:
                        etf_reasoning.append(f"Negative signal from {agent_display}")
                    else:
                        etf_reasoning.append(f"Neutral signal from {agent_display}")
            
            if etf_reasoning:
                print(f"   • {'; '.join(etf_reasoning[:3])}")  # Show top 3 reasons
            else:
                print(f"   • Recommended based on macro trends and risk assessment")
            print()
        
        # Portfolio characteristics
        print("📊 Portfolio Characteristics:")
        total_allocation = sum(allocations.values())
        print(f"   • Total Allocation: {total_allocation:.1f}%")
        print(f"   • Number of ETFs: {len(allocations)}")
        print(f"   • Diversification: {'High' if len(allocations) >= 5 else 'Moderate' if len(allocations) >= 3 else 'Low'}")
        
        # Risk assessment
        high_risk_etfs = [etf for etf, alloc in allocations.items() if alloc > 30]
        if high_risk_etfs:
            print(f"   • Concentration Risk: High allocation to {', '.join(high_risk_etfs)}")
        else:
            print(f"   • Concentration Risk: Well-diversified portfolio")
    
    print("\n" + "="*80)
    print("📋 ANALYSIS COMPLETE - USE THIS REPORT FOR INVESTMENT DECISIONS")
    print("="*80)
    
    return complete_state  # Return state for file saving


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


def save_comprehensive_report(complete_state, allocations, universe, date, duration):
    """Save the comprehensive analysis report to a file."""
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"macro_analysis_report_{timestamp}.txt"
    filepath = os.path.join(reports_dir, filename)
    
    # Capture all the detailed analysis
    agent_reasoning = complete_state.get('agent_reasoning', {})
    macro_data = complete_state.get('macro_data', {})
    news_data = complete_state.get('news', [])
    debate_output = complete_state.get('debate_output', [])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GLOBAL MACRO ETF TRADING SYSTEM - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {date}\n")
        f.write(f"ETF Universe: {', '.join(universe)}\n")
        f.write(f"Analysis Duration: {duration:.1f} seconds\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Final allocations
        f.write("📊 FINAL MACRO ALLOCATIONS (Buy Percentages):\n")
        f.write("-" * 50 + "\n")
        if allocations:
            sorted_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
            for i, (etf, allocation) in enumerate(sorted_allocations, 1):
                f.write(f"{i:2d}. {etf:6s}: {allocation:6.1f}%\n")
            
            total_allocation = sum(allocations.values())
            f.write("-" * 50 + "\n")
            f.write(f"Total: {total_allocation:.1f}%\n")
            f.write("\n")
        
        # 1. MACRO ECONOMIC ANALYSIS
        f.write("🏛️  MACRO ECONOMIC ANALYSIS\n")
        f.write("="*50 + "\n")
        
        # Show ETF data lookback information
        etf_data = complete_state.get('etf_data', pd.DataFrame())
        if not etf_data.empty:
            # Calculate actual lookback period
            if hasattr(etf_data.index, 'min') and hasattr(etf_data.index, 'max'):
                start_date = etf_data.index.min()
                end_date = etf_data.index.max()
                years_of_data = (end_date - start_date).days / 365.25
                f.write(f"📊 ETF Historical Data:\n")
                f.write(f"  • Lookback Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
                f.write(f"  • Years of Data: {years_of_data:.1f} years\n")
                f.write(f"  • ETFs Analyzed: {len(etf_data.columns.get_level_values(0).unique()) if isinstance(etf_data.columns, pd.MultiIndex) else len(etf_data.columns)}\n")
                f.write(f"  • Data Points: {len(etf_data)} trading days\n")
            else:
                f.write(f"📊 ETF Historical Data: {len(etf_data)} data points available\n")
        else:
            f.write("📊 ETF Historical Data: No data available\n")
        
        if macro_data:
            f.write("\n📈 Key Economic Indicators:\n")
            for indicator, data in macro_data.items():
                if 'error' not in data and data.get('latest_value') is not None:
                    latest_value = data.get('latest_value', 'N/A')
                    periods = data.get('periods', 0)
                    f.write(f"  • {indicator}: {latest_value} ({periods} periods of data)\n")
                else:
                    f.write(f"  • {indicator}: Error - {data.get('error', 'No data')}\n")
        else:
            f.write("  No macro economic data available\n")
        
        # Macro economist reasoning
        if 'macro_economist' in agent_reasoning:
            macro_agent = agent_reasoning['macro_economist']
            f.write(f"\n🎯 Macro Economist Assessment:\n")
            f.write(f"  Reasoning: {macro_agent.get('reasoning', 'No detailed reasoning provided')}\n")
            if macro_agent.get('key_factors'):
                f.write("  Key Macro Factors:\n")
                for factor in macro_agent['key_factors']:
                    f.write(f"    • {factor}\n")
            if macro_agent.get('scores'):
                f.write("  ETF Scores with Macro Analysis:\n")
                for etf, score in macro_agent['scores'].items():
                    sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                    f.write(f"    {etf}: {score:.2f} {sentiment}\n")
        
        # 2. GEOPOLITICAL ANALYSIS
        f.write("\n🌍 GEOPOLITICAL ANALYSIS\n")
        f.write("="*50 + "\n")
        if news_data:
            f.write("📰 Key News Events:\n")
            for i, article in enumerate(news_data, 1):
                title = article.get('title', 'Unknown Title')
                summary = article.get('summary', 'No summary available')
                sentiment = article.get('sentiment', 'neutral')
                f.write(f"  {i}. {title}\n")
                f.write(f"     Summary: {summary[:200]}...\n")
                f.write(f"     Sentiment: {sentiment}\n")
                f.write(f"     Impact: This news affects global markets and specific regions\n")
                f.write("\n")
        else:
            f.write("  No geopolitical news data available\n")
        
        # Geopolitical analyst reasoning
        if 'geopolitical_analyst' in agent_reasoning:
            geo_agent = agent_reasoning['geopolitical_analyst']
            f.write(f"\n🎯 Geopolitical Analyst Assessment:\n")
            f.write(f"  Reasoning: {geo_agent.get('reasoning', 'No detailed reasoning provided')}\n")
            if geo_agent.get('key_factors'):
                f.write("  Key Geopolitical Factors:\n")
                for factor in geo_agent['key_factors']:
                    f.write(f"    • {factor}\n")
            if geo_agent.get('scores'):
                f.write("  ETF Scores with Geopolitical Analysis:\n")
                for etf, score in geo_agent['scores'].items():
                    sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                    f.write(f"    {etf}: {score:.2f} {sentiment}\n")
        
        # 3. CORRELATION ANALYSIS
        f.write("\n📈 CORRELATION & DIVERSIFICATION ANALYSIS\n")
        f.write("="*50 + "\n")
        if 'correlation_specialist' in agent_reasoning:
            corr_agent = agent_reasoning['correlation_specialist']
            f.write(f"🎯 Correlation Specialist Assessment:\n")
            f.write(f"  Reasoning: {corr_agent.get('reasoning', 'No detailed reasoning provided')}\n")
            if corr_agent.get('scores'):
                f.write("  Diversification Scores:\n")
                for etf, score in corr_agent['scores'].items():
                    diversification = "🟢 High Diversification" if score > 0.3 else "🔴 Low Diversification" if score < -0.3 else "⚪ Moderate"
                    f.write(f"    {etf}: {score:.2f} {diversification}\n")
        
        # 4. DEBATE ANALYSIS
        f.write("\n⚔️  BULLISH vs BEARISH DEBATE ANALYSIS\n")
        f.write("="*50 + "\n")
        if debate_output:
            f.write("🗣️  Full Debate Content:\n")
            for i, line in enumerate(debate_output, 1):
                if line.strip():
                    f.write(f"  {i:2d}. {line}\n")
        else:
            f.write("  No debate data available\n")
        
        # 5. RISK ASSESSMENT
        f.write("\n🛡️  RISK ASSESSMENT\n")
        f.write("="*50 + "\n")
        if 'risk_manager' in agent_reasoning:
            risk_agent = agent_reasoning['risk_manager']
            f.write(f"🎯 Risk Manager Assessment:\n")
            f.write(f"  Reasoning: {risk_agent.get('reasoning', 'No detailed reasoning provided')}\n")
            if risk_agent.get('risk_factors'):
                f.write("  Key Risk Factors:\n")
                for factor in risk_agent['risk_factors']:
                    f.write(f"    • {factor}\n")
            if risk_agent.get('adjustments'):
                f.write("  Risk Adjustments Applied:\n")
                for etf, adjustment in risk_agent['adjustments'].items():
                    f.write(f"    {etf}: {adjustment}\n")
        
        # 6. PORTFOLIO OPTIMIZATION
        f.write("\n⚖️  PORTFOLIO OPTIMIZATION\n")
        f.write("="*50 + "\n")
        if 'portfolio_optimizer' in agent_reasoning:
            opt_agent = agent_reasoning['portfolio_optimizer']
            f.write(f"🎯 Portfolio Optimizer Assessment:\n")
            f.write(f"  Method: {opt_agent.get('optimization_method', 'Not specified')}\n")
            f.write(f"  Reasoning: {opt_agent.get('reasoning', 'No detailed reasoning provided')}\n")
            if opt_agent.get('performance_metrics'):
                f.write("  Performance Metrics:\n")
                for metric, value in opt_agent['performance_metrics'].items():
                    f.write(f"    {metric}: {value}\n")
        
        # 7. COMPREHENSIVE ETF ANALYSIS
        f.write("\n🎯 COMPREHENSIVE ETF ANALYSIS\n")
        f.write("="*50 + "\n")
        if allocations:
            # Get all scores for comprehensive analysis
            all_scores = {}
            for agent_name, agent_data in agent_reasoning.items():
                if agent_data.get('scores'):
                    all_scores[agent_name] = agent_data['scores']
            
            f.write("📊 ETF-by-ETF Analysis:\n")
            for etf in sorted(allocations.keys(), key=lambda x: allocations[x], reverse=True):
                allocation = allocations[etf]
                f.write(f"\n  📈 {etf} ({allocation:.1f}% allocation):\n")
                
                # Collect all scores for this ETF
                etf_scores = {}
                for agent_name, scores in all_scores.items():
                    if etf in scores:
                        etf_scores[agent_name] = scores[etf]
                
                # Display scores
                if etf_scores:
                    f.write("    Scores by Analyst:\n")
                    for agent_name, score in etf_scores.items():
                        agent_display = agent_name.replace('_', ' ').title()
                        sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
                        f.write(f"      {agent_display}: {score:.2f} {sentiment}\n")
                
                # Provide rationale for allocation
                if allocation > 20:
                    f.write(f"    💡 HIGH ALLOCATION: {etf} receives {allocation:.1f}% due to strong positive signals across multiple analysts\n")
                elif allocation > 10:
                    f.write(f"    💡 MODERATE ALLOCATION: {etf} receives {allocation:.1f}% with mixed but generally positive signals\n")
                elif allocation > 5:
                    f.write(f"    💡 LOW ALLOCATION: {etf} receives {allocation:.1f}% with cautious positioning\n")
                else:
                    f.write(f"    💡 MINIMAL ALLOCATION: {etf} receives {allocation:.1f}% due to negative signals or high risk\n")
        
        # 8. FINAL RECOMMENDATIONS
        f.write("\n🎯 FINAL RECOMMENDATIONS & RATIONALE\n")
        f.write("="*50 + "\n")
        if allocations:
            f.write("📋 Portfolio Allocation Rationale:\n")
            f.write("\n")
            
            # Sort ETFs by allocation
            sorted_etfs = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
            
            for i, (etf, allocation) in enumerate(sorted_etfs, 1):
                f.write(f"{i}. {etf} ({allocation:.1f}%):\n")
                
                # Get reasoning for this ETF
                etf_reasoning = []
                for agent_name, agent_data in agent_reasoning.items():
                    if agent_data.get('scores') and etf in agent_data['scores']:
                        score = agent_data['scores'][etf]
                        agent_display = agent_name.replace('_', ' ').title()
                        if score > 0.1:
                            etf_reasoning.append(f"Strong positive signal from {agent_display}")
                        elif score < -0.1:
                            etf_reasoning.append(f"Negative signal from {agent_display}")
                        else:
                            etf_reasoning.append(f"Neutral signal from {agent_display}")
                
                if etf_reasoning:
                    f.write(f"   • {'; '.join(etf_reasoning[:3])}\n")  # Show top 3 reasons
                else:
                    f.write(f"   • Recommended based on macro trends and risk assessment\n")
                f.write("\n")
            
            # Portfolio characteristics
            f.write("📊 Portfolio Characteristics:\n")
            total_allocation = sum(allocations.values())
            f.write(f"   • Total Allocation: {total_allocation:.1f}%\n")
            f.write(f"   • Number of ETFs: {len(allocations)}\n")
            f.write(f"   • Diversification: {'High' if len(allocations) >= 5 else 'Moderate' if len(allocations) >= 3 else 'Low'}\n")
            
            # Risk assessment
            high_risk_etfs = [etf for etf, alloc in allocations.items() if alloc > 30]
            if high_risk_etfs:
                f.write(f"   • Concentration Risk: High allocation to {', '.join(high_risk_etfs)}\n")
            else:
                f.write(f"   • Concentration Risk: Well-diversified portfolio\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("📋 ANALYSIS COMPLETE - USE THIS REPORT FOR INVESTMENT DECISIONS\n")
        f.write("="*80 + "\n")
    
    return filepath


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
        
        # Save comprehensive report to file
        report_file = save_comprehensive_report(complete_state, allocations, args.universe, args.date, duration)
        print(f"\n📄 Comprehensive report saved to: {report_file}")
        
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

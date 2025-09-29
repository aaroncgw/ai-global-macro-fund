"""
Test All Macro Analyst Agents

This module tests all three macro analyst agents working together
to demonstrate the complete analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.geopolitical_analyst import GeopoliticalAnalystAgent
from src.agents.correlation_specialist import CorrelationSpecialistAgent


def create_sample_data():
    """Create sample data for testing all agents."""
    
    # Create sample ETF data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    etf_data = pd.DataFrame({
        'SPY': 100 + np.cumsum(np.random.randn(252) * 0.01),
        'QQQ': 200 + np.cumsum(np.random.randn(252) * 0.015),
        'TLT': 150 + np.cumsum(np.random.randn(252) * 0.005),
        'GLD': 180 + np.cumsum(np.random.randn(252) * 0.008),
        'EWJ': 120 + np.cumsum(np.random.randn(252) * 0.012),
        'FXI': 80 + np.cumsum(np.random.randn(252) * 0.018),
        'UUP': 25 + np.cumsum(np.random.randn(252) * 0.003),
        'FXE': 110 + np.cumsum(np.random.randn(252) * 0.004)
    }, index=dates)
    
    # Create sample macro data
    macro_data = {
        'CPIAUCSL': {
            'latest_value': 300.0,
            'periods': 12,
            'trend': 'increasing'
        },
        'UNRATE': {
            'latest_value': 3.5,
            'periods': 12,
            'trend': 'stable'
        },
        'FEDFUNDS': {
            'latest_value': 5.25,
            'periods': 12,
            'trend': 'increasing'
        },
        'GDPC1': {
            'latest_value': 2.1,
            'periods': 4,
            'trend': 'moderate_growth'
        }
    }
    
    # Create sample news data
    news_data = [
        {
            'title': 'Federal Reserve Maintains Hawkish Stance on Interest Rates',
            'summary': 'Fed signals continued rate hikes to combat inflation, affecting bond markets and currency valuations',
            'sentiment': 'negative',
            'impact': 'high'
        },
        {
            'title': 'China-US Trade Tensions Escalate',
            'summary': 'New tariffs announced on technology exports, impacting global supply chains and emerging markets',
            'sentiment': 'negative',
            'impact': 'medium'
        },
        {
            'title': 'European Central Bank Signals Dovish Pivot',
            'summary': 'ECB hints at potential rate cuts to support economic recovery, boosting European equities',
            'sentiment': 'positive',
            'impact': 'medium'
        },
        {
            'title': 'Geopolitical Tensions Rise in Middle East',
            'summary': 'Regional conflicts escalate, driving safe-haven demand for gold and US Treasury bonds',
            'sentiment': 'negative',
            'impact': 'high'
        }
    ]
    
    return etf_data, macro_data, news_data


def test_all_agents():
    """Test all three macro analyst agents."""
    
    print("Testing All Macro Analyst Agents")
    print("="*50)
    
    try:
        # Create sample data
        etf_data, macro_data, news_data = create_sample_data()
        universe = ['SPY', 'QQQ', 'TLT', 'GLD', 'EWJ', 'FXI', 'UUP', 'FXE']
        
        # Initialize all agents
        macro_economist = MacroEconomistAgent("MacroEconomist")
        geopolitical_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        correlation_specialist = CorrelationSpecialistAgent("CorrelationSpecialist")
        
        print("✓ All agents initialized successfully")
        
        # Create initial state
        state = {
            'macro_data': macro_data,
            'etf_data': etf_data,
            'news': news_data,
            'universe': universe,
            'analyst_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Initial state created with {len(universe)} ETFs")
        print(f"  ETFs: {', '.join(universe)}")
        print(f"  Macro indicators: {len(macro_data)}")
        print(f"  News articles: {len(news_data)}")
        
        # Run macro economist analysis
        print("\n1. Running Macro Economist Analysis...")
        state = macro_economist.analyze(state)
        macro_scores = state['analyst_scores']['macro']
        print(f"   Macro scores: {macro_scores}")
        
        # Run geopolitical analyst analysis
        print("\n2. Running Geopolitical Analyst Analysis...")
        state = geopolitical_analyst.analyze(state)
        geo_scores = state['analyst_scores']['geo']
        print(f"   Geopolitical scores: {geo_scores}")
        
        # Run correlation specialist analysis
        print("\n3. Running Correlation Specialist Analysis...")
        state = correlation_specialist.analyze(state)
        corr_scores = state['analyst_scores']['correlation']
        print(f"   Correlation scores: {corr_scores}")
        
        # Combine and analyze results
        print("\n4. Combined Analysis Results:")
        print("="*30)
        
        all_scores = {
            'macro': macro_scores,
            'geopolitical': geo_scores,
            'correlation': corr_scores
        }
        
        # Calculate average scores
        avg_scores = {}
        for etf in universe:
            scores = [all_scores[analyst][etf] for analyst in all_scores.keys()]
            avg_scores[etf] = sum(scores) / len(scores)
        
        # Sort by average score
        sorted_etfs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("ETF Rankings (by average score):")
        for i, (etf, score) in enumerate(sorted_etfs, 1):
            print(f"  {i}. {etf}: {score:.3f}")
        
        # Show detailed breakdown
        print("\nDetailed Score Breakdown:")
        for etf in universe:
            print(f"\n{etf}:")
            print(f"  Macro: {macro_scores.get(etf, 0.0):.3f}")
            print(f"  Geopolitical: {geo_scores.get(etf, 0.0):.3f}")
            print(f"  Correlation: {corr_scores.get(etf, 0.0):.3f}")
            print(f"  Average: {avg_scores.get(etf, 0.0):.3f}")
        
        # Show top recommendations
        print("\nTop Recommendations:")
        top_3 = sorted_etfs[:3]
        for i, (etf, score) in enumerate(top_3, 1):
            print(f"  {i}. {etf} (Score: {score:.3f})")
        
        print("\nBottom Recommendations:")
        bottom_3 = sorted_etfs[-3:]
        for i, (etf, score) in enumerate(bottom_3, 1):
            print(f"  {i}. {etf} (Score: {score:.3f})")
        
        print("\n✓ All agent tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_agent_integration():
    """Test how agents work together in a LangGraph-style workflow."""
    
    print("\nTesting Agent Integration")
    print("="*30)
    
    try:
        # Create sample data
        etf_data, macro_data, news_data = create_sample_data()
        universe = ['SPY', 'QQQ', 'TLT', 'GLD']
        
        # Initialize agents
        macro_economist = MacroEconomistAgent("MacroEconomist")
        geopolitical_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        correlation_specialist = CorrelationSpecialistAgent("CorrelationSpecialist")
        
        # Simulate LangGraph workflow
        state = {
            'macro_data': macro_data,
            'etf_data': etf_data,
            'news': news_data,
            'universe': universe,
            'analyst_scores': {},
            'workflow_step': 0
        }
        
        print("Simulating LangGraph workflow...")
        
        # Step 1: Macro analysis
        state['workflow_step'] = 1
        state = macro_economist.analyze(state)
        print(f"Step 1: Macro analysis completed")
        
        # Step 2: Geopolitical analysis
        state['workflow_step'] = 2
        state = geopolitical_analyst.analyze(state)
        print(f"Step 2: Geopolitical analysis completed")
        
        # Step 3: Correlation analysis
        state['workflow_step'] = 3
        state = correlation_specialist.analyze(state)
        print(f"Step 3: Correlation analysis completed")
        
        # Final state
        print(f"\nFinal state contains:")
        print(f"  - Workflow step: {state['workflow_step']}")
        print(f"  - Analyst scores: {len(state['analyst_scores'])} analysts")
        print(f"  - Universe: {len(state['universe'])} ETFs")
        
        print("✓ Agent integration test completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Macro Analyst Agents Test Suite")
    print("="*50)
    
    # Test individual agents
    test_all_agents()
    
    # Test agent integration
    test_agent_integration()
    
    print("\n" + "="*50)
    print("All tests completed!")

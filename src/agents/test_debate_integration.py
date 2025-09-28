"""
Test Debate Integration with Macro Analyst Agents

This module tests the complete integration of macro analyst agents
with debate researchers to demonstrate the full analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.geopolitical_analyst import GeopoliticalAnalystAgent
from src.agents.correlation_specialist import CorrelationSpecialistAgent
from src.agents.debate_researchers import debate, analyze_debate_results


def create_comprehensive_sample_data():
    """Create comprehensive sample data for testing the full pipeline."""
    
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
        'FXE': 110 + np.cumsum(np.random.randn(252) * 0.004),
        'EWG': 130 + np.cumsum(np.random.randn(252) * 0.010),
        'INDA': 90 + np.cumsum(np.random.randn(252) * 0.014)
    }, index=dates)
    
    # Create sample macro data
    macro_data = {
        'CPIAUCSL': {
            'latest_value': 300.0,
            'periods': 12,
            'trend': 'increasing',
            'description': 'Consumer Price Index'
        },
        'UNRATE': {
            'latest_value': 3.5,
            'periods': 12,
            'trend': 'stable',
            'description': 'Unemployment Rate'
        },
        'FEDFUNDS': {
            'latest_value': 5.25,
            'periods': 12,
            'trend': 'increasing',
            'description': 'Federal Funds Rate'
        },
        'GDPC1': {
            'latest_value': 2.1,
            'periods': 4,
            'trend': 'moderate_growth',
            'description': 'Real GDP Growth'
        }
    }
    
    # Create sample news data
    news_data = [
        {
            'title': 'Federal Reserve Signals Hawkish Stance on Interest Rates',
            'summary': 'Fed officials indicate continued rate hikes to combat persistent inflation, affecting bond markets and currency valuations globally',
            'sentiment': 'negative',
            'impact': 'high',
            'region': 'US'
        },
        {
            'title': 'China-US Trade Tensions Escalate Over Technology Exports',
            'summary': 'New tariffs announced on semiconductor exports, impacting global supply chains and emerging market equities',
            'sentiment': 'negative',
            'impact': 'high',
            'region': 'Global'
        },
        {
            'title': 'European Central Bank Signals Dovish Pivot',
            'summary': 'ECB hints at potential rate cuts to support economic recovery, boosting European equities and weakening Euro',
            'sentiment': 'positive',
            'impact': 'medium',
            'region': 'Europe'
        },
        {
            'title': 'Geopolitical Tensions Rise in Middle East',
            'summary': 'Regional conflicts escalate, driving safe-haven demand for gold and US Treasury bonds while pressuring oil prices',
            'sentiment': 'negative',
            'impact': 'high',
            'region': 'Middle East'
        },
        {
            'title': 'Japan Implements Aggressive Monetary Stimulus',
            'summary': 'Bank of Japan announces additional quantitative easing measures, supporting Japanese equities and weakening Yen',
            'sentiment': 'positive',
            'impact': 'medium',
            'region': 'Japan'
        }
    ]
    
    return etf_data, macro_data, news_data


def test_complete_analysis_pipeline():
    """Test the complete analysis pipeline from agents to debate."""
    
    print("Complete Analysis Pipeline Test")
    print("="*50)
    
    try:
        # Create sample data
        etf_data, macro_data, news_data = create_comprehensive_sample_data()
        universe = ['SPY', 'QQQ', 'TLT', 'GLD', 'EWJ', 'FXI', 'UUP', 'FXE', 'EWG', 'INDA']
        
        print(f"✓ Sample data created")
        print(f"  ETFs: {len(universe)}")
        print(f"  Macro indicators: {len(macro_data)}")
        print(f"  News articles: {len(news_data)}")
        
        # Initialize all agents
        macro_economist = MacroEconomistAgent("MacroEconomist")
        geopolitical_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        correlation_specialist = CorrelationSpecialistAgent("CorrelationSpecialist")
        
        print(f"✓ All agents initialized")
        
        # Create initial state
        state = {
            'macro_data': macro_data,
            'etf_data': etf_data,
            'news': news_data,
            'universe': universe,
            'analyst_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Run macro economist analysis
        print("\n1. Running Macro Economist Analysis...")
        state = macro_economist.analyze(state)
        macro_scores = state['analyst_scores']['macro']
        print(f"   Macro scores generated for {len(macro_scores)} ETFs")
        
        # Step 2: Run geopolitical analyst analysis
        print("\n2. Running Geopolitical Analyst Analysis...")
        state = geopolitical_analyst.analyze(state)
        geo_scores = state['analyst_scores']['geo']
        print(f"   Geopolitical scores generated for {len(geo_scores)} ETFs")
        
        # Step 3: Run correlation specialist analysis
        print("\n3. Running Correlation Specialist Analysis...")
        state = correlation_specialist.analyze(state)
        corr_scores = state['analyst_scores']['correlation']
        print(f"   Correlation scores generated for {len(corr_scores)} ETFs")
        
        # Step 4: Run debate
        print("\n4. Running Macro Debate...")
        state = debate(state, rounds=2)
        debate_output = state.get('debate_output', [])
        print(f"   Debate completed with {len(debate_output)} output lines")
        
        # Step 5: Analyze debate results
        print("\n5. Analyzing Debate Results...")
        state = analyze_debate_results(state)
        debate_analysis = state.get('debate_analysis', '')
        print(f"   Debate analysis completed ({len(debate_analysis)} characters)")
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        # Show analyst scores
        print("\nAnalyst Scores Summary:")
        for analyst, scores in state['analyst_scores'].items():
            print(f"\n{analyst.upper()} SCORES:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for etf, score in sorted_scores:
                print(f"  {etf}: {score:.3f}")
        
        # Show debate output
        print(f"\nDebate Output (first 5 lines):")
        for line in debate_output[:5]:
            print(f"  {line}")
        
        # Show debate analysis
        print(f"\nDebate Analysis (first 200 characters):")
        print(f"  {debate_analysis[:200]}...")
        
        print("\n✓ Complete analysis pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def test_debate_scenarios():
    """Test different debate scenarios with varying analyst scores."""
    
    print("\nDebate Scenarios Test")
    print("="*30)
    
    try:
        # Scenario 1: Bullish macro environment
        print("\nScenario 1: Bullish Macro Environment")
        bullish_state = {
            'analyst_scores': {
                'macro': {'SPY': 0.8, 'QQQ': 0.9, 'TLT': -0.2, 'GLD': 0.6},
                'geo': {'SPY': 0.7, 'QQQ': 0.8, 'TLT': 0.3, 'GLD': 0.5},
                'correlation': {'SPY': 0.4, 'QQQ': 0.3, 'TLT': 0.6, 'GLD': 0.7}
            },
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'debate_output': []
        }
        
        result1 = debate(bullish_state, rounds=1)
        print(f"   Bullish scenario debate completed")
        
        # Scenario 2: Bearish macro environment
        print("\nScenario 2: Bearish Macro Environment")
        bearish_state = {
            'analyst_scores': {
                'macro': {'SPY': -0.8, 'QQQ': -0.9, 'TLT': 0.2, 'GLD': 0.6},
                'geo': {'SPY': -0.7, 'QQQ': -0.8, 'TLT': 0.3, 'GLD': 0.5},
                'correlation': {'SPY': -0.4, 'QQQ': -0.3, 'TLT': 0.6, 'GLD': 0.7}
            },
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'debate_output': []
        }
        
        result2 = debate(bearish_state, rounds=1)
        print(f"   Bearish scenario debate completed")
        
        # Scenario 3: Mixed signals
        print("\nScenario 3: Mixed Signals")
        mixed_state = {
            'analyst_scores': {
                'macro': {'SPY': 0.2, 'QQQ': -0.3, 'TLT': -0.8, 'GLD': 0.7},
                'geo': {'SPY': -0.4, 'QQQ': 0.1, 'TLT': 0.6, 'GLD': 0.8},
                'correlation': {'SPY': 0.1, 'QQQ': 0.0, 'TLT': 0.5, 'GLD': 0.6}
            },
            'universe': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'debate_output': []
        }
        
        result3 = debate(mixed_state, rounds=1)
        print(f"   Mixed signals scenario debate completed")
        
        print("\n✓ All debate scenarios completed successfully!")
        
    except Exception as e:
        print(f"❌ Debate scenarios test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Debate Integration Test Suite")
    print("="*50)
    
    # Test complete pipeline
    test_complete_analysis_pipeline()
    
    # Test debate scenarios
    test_debate_scenarios()
    
    print("\n" + "="*50)
    print("All tests completed!")

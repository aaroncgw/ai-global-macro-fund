"""
Test All Revamped Macro Analyst Agents

This module tests all four revamped macro analyst agents working together
to demonstrate the complete analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.geopolitical_analyst import GeopoliticalAnalystAgent
from src.agents.risk_manager import RiskManager
from src.agents.portfolio_manager import PortfolioManagerAgent


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
    """Test all four revamped macro analyst agents."""
    
    print("Testing All Revamped Macro Analyst Agents")
    print("="*50)
    
    try:
        # Create sample data
        etf_data, macro_data, news_data = create_sample_data()
        universe = ['SPY', 'TLT', 'GLD', 'EWJ', 'FXI']
        
        # Initialize all agents
        macro_economist = MacroEconomistAgent("MacroEconomist")
        geopolitical_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        risk_manager = RiskManager("RiskManager")
        portfolio_manager = PortfolioManagerAgent("PortfolioManagerAgent")
        
        print("✓ All agents initialized successfully")
        
        # Create initial state
        state = {
            'macro_data': macro_data,
            'etf_data': etf_data,
            'news': news_data,
            'universe': universe,
            'agent_reasoning': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Initial state created with {len(universe)} ETFs")
        print(f"  ETFs: {', '.join(universe)}")
        print(f"  Macro indicators: {len(macro_data)}")
        print(f"  News articles: {len(news_data)}")
        
        # Run macro economist analysis
        print("\n1. Running Macro Economist Analysis...")
        state = macro_economist.analyze(state)
        macro_scores = state.get('macro_scores', {})
        print(f"   Macro scores generated for {len(macro_scores)} ETFs")
        
        # Run geopolitical analyst analysis
        print("\n2. Running Geopolitical Analyst Analysis...")
        state = geopolitical_analyst.analyze(state)
        geo_scores = state.get('geo_scores', {})
        print(f"   Geopolitical scores generated for {len(geo_scores)} ETFs")
        
        # Run risk manager analysis
        print("\n3. Running Risk Manager Analysis...")
        state = risk_manager.assess(state)
        risk_assessments = state.get('risk_assessments', {})
        print(f"   Risk assessments generated for {len(risk_assessments)} ETFs")
        
        # Run portfolio agent analysis
        print("\n4. Running Portfolio Agent Analysis...")
        state = portfolio_manager.manage(state)
        final_allocations = state.get('final_allocations', {})
        print(f"   Final allocations generated for {len(final_allocations)} ETFs")
        
        # Display results
        print("\n5. Final Results:")
        print("="*30)
        
        print("Final Portfolio Allocations:")
        for etf, allocation in final_allocations.items():
            if isinstance(allocation, dict):
                action = allocation.get('action', 'unknown')
                allocation_pct = allocation.get('allocation', 0.0)
                reason = allocation.get('reason', 'No reasoning provided')
                print(f"  {etf}: {action.upper()} {allocation_pct:.1%} - {reason[:50]}...")
            else:
                print(f"  {etf}: {allocation}")
        
        # Show agent reasoning summary
        print("\nAgent Reasoning Summary:")
        agent_reasoning = state.get('agent_reasoning', {})
        for agent_name, reasoning in agent_reasoning.items():
            print(f"  {agent_name}: {reasoning.get('reasoning', 'No reasoning available')}")
        
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
        universe = ['SPY', 'TLT', 'GLD']
        
        # Initialize agents
        macro_economist = MacroEconomistAgent("MacroEconomist")
        geopolitical_analyst = GeopoliticalAnalystAgent("GeopoliticalAnalyst")
        risk_manager = RiskManager("RiskManager")
        portfolio_manager = PortfolioManagerAgent("PortfolioManagerAgent")
        
        # Simulate LangGraph workflow
        state = {
            'macro_data': macro_data,
            'etf_data': etf_data,
            'news': news_data,
            'universe': universe,
            'agent_reasoning': {},
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
        
        # Step 3: Risk management
        state['workflow_step'] = 3
        state = risk_manager.assess(state)
        print(f"Step 3: Risk management completed")
        
        # Step 4: Portfolio optimization
        state['workflow_step'] = 4
        state = portfolio_manager.manage(state)
        print(f"Step 4: Portfolio optimization completed")
        
        # Final state
        print(f"\nFinal state contains:")
        print(f"  - Workflow step: {state['workflow_step']}")
        print(f"  - Agent reasoning: {len(state.get('agent_reasoning', {}))} agents")
        print(f"  - Universe: {len(state['universe'])} ETFs")
        print(f"  - Final allocations: {len(state.get('final_allocations', {}))} ETFs")
        
        print("✓ Agent integration test completed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Revamped Macro Analyst Agents Test Suite")
    print("="*50)
    
    # Test individual agents
    test_all_agents()
    
    # Test agent integration
    test_agent_integration()
    
    print("\n" + "="*50)
    print("All tests completed!")

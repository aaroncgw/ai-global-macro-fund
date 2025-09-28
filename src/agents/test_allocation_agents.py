"""
Test Allocation Agents Integration

This module tests the complete allocation pipeline from trader to risk manager
to portfolio optimizer for ETF allocation decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.agents.trader_agent import TraderAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.portfolio_optimizer import PortfolioOptimizerAgent


def create_allocation_sample_data():
    """Create sample data for testing the allocation pipeline."""
    
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
            'trend': 'increasing',
            'description': 'Consumer Price Index'
        },
        'UNRATE': {
            'latest_value': 3.5,
            'trend': 'stable',
            'description': 'Unemployment Rate'
        },
        'FEDFUNDS': {
            'latest_value': 5.25,
            'trend': 'increasing',
            'description': 'Federal Funds Rate'
        },
        'GDPC1': {
            'latest_value': 2.1,
            'trend': 'moderate_growth',
            'description': 'Real GDP Growth'
        }
    }
    
    # Create sample debate results
    debate_output = [
        '=== MACRO ETF DEBATE STARTED ===',
        'Universe: SPY, QQQ, TLT, GLD, EWJ, FXI, UUP, FXE',
        'Analysts: macro, geo, correlation',
        '=== ROUND 1 ===',
        'BULLISH ARGUMENT: Strong macro trends support growth assets with inflation hedges...',
        'BEARISH COUNTER-ARGUMENT: Geopolitical risks create headwinds for international exposure...',
        '=== ROUND 2 ===',
        'BULLISH ARGUMENT: Central bank policies remain supportive of risk assets...',
        'BEARISH COUNTER-ARGUMENT: Rising rates create headwinds for bond allocations...'
    ]
    
    # Create sample analyst scores
    analyst_scores = {
        'macro': {
            'SPY': 0.4, 'QQQ': 0.6, 'TLT': -0.8, 'GLD': 0.7,
            'EWJ': -0.3, 'FXI': -0.5, 'UUP': 0.5, 'FXE': -0.4
        },
        'geo': {
            'SPY': -0.4, 'QQQ': -0.6, 'TLT': 0.3, 'GLD': 0.7,
            'EWJ': 0.5, 'FXI': -0.8, 'UUP': 0.6, 'FXE': 0.2
        },
        'correlation': {
            'SPY': 0.1, 'QQQ': 0.0, 'TLT': 0.6, 'GLD': 0.7,
            'EWJ': 0.2, 'FXI': 0.8, 'UUP': 0.5, 'FXE': 0.4
        }
    }
    
    return etf_data, macro_data, debate_output, analyst_scores


def test_complete_allocation_pipeline():
    """Test the complete allocation pipeline from trader to optimizer."""
    
    print("Complete Allocation Pipeline Test")
    print("="*50)
    
    try:
        # Create sample data
        etf_data, macro_data, debate_output, analyst_scores = create_allocation_sample_data()
        universe = ['SPY', 'QQQ', 'TLT', 'GLD', 'EWJ', 'FXI', 'UUP', 'FXE']
        
        print(f"✓ Sample data created")
        print(f"  ETFs: {len(universe)}")
        print(f"  Macro indicators: {len(macro_data)}")
        print(f"  Debate rounds: {len([line for line in debate_output if 'ROUND' in line])}")
        
        # Initialize all agents
        trader = TraderAgent("Trader")
        risk_manager = RiskManagerAgent("RiskManager")
        optimizer = PortfolioOptimizerAgent("Optimizer")
        
        print(f"✓ All agents initialized")
        
        # Create initial state
        state = {
            'debate_output': debate_output,
            'analyst_scores': analyst_scores,
            'macro_data': macro_data,
            'etf_data': etf_data,
            'universe': universe,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Trader proposes initial allocations
        print("\n1. Running Trader Agent...")
        state = trader.propose(state)
        proposed_allocations = state['proposed_allocations']
        print(f"   Proposed allocations: {proposed_allocations}")
        
        # Step 2: Risk Manager adjusts for risk
        print("\n2. Running Risk Manager...")
        state['risk_factors'] = {
            'volatility_regime': 'high',
            'geopolitical_risk': 'elevated',
            'liquidity_conditions': 'normal'
        }
        state = risk_manager.assess(state)
        risk_adjusted_allocations = state['risk_adjusted_allocations']
        print(f"   Risk-adjusted allocations: {risk_adjusted_allocations}")
        
        # Step 3: Portfolio Optimizer optimizes
        print("\n3. Running Portfolio Optimizer...")
        state = optimizer.optimize(state)
        final_allocations = state['final_allocations']
        print(f"   Final optimized allocations: {final_allocations}")
        
        # Display results
        print("\n" + "="*50)
        print("ALLOCATION RESULTS")
        print("="*50)
        
        # Show allocation progression
        print("\nAllocation Progression:")
        print("="*30)
        
        for etf in universe:
            proposed = proposed_allocations.get(etf, 0.0)
            risk_adj = risk_adjusted_allocations.get(etf, 0.0)
            final = final_allocations.get(etf, 0.0)
            
            print(f"{etf}:")
            print(f"  Proposed: {proposed:.1f}%")
            print(f"  Risk-Adj: {risk_adj:.1f}%")
            print(f"  Final:    {final:.1f}%")
            print()
        
        # Show allocation changes
        print("Allocation Changes:")
        print("="*20)
        
        for etf in universe:
            proposed = proposed_allocations.get(etf, 0.0)
            final = final_allocations.get(etf, 0.0)
            change = final - proposed
            
            if abs(change) > 1.0:  # Only show significant changes
                print(f"{etf}: {proposed:.1f}% → {final:.1f}% ({change:+.1f}%)")
        
        # Show top allocations
        print("\nTop Allocations (Final):")
        print("="*25)
        
        sorted_allocations = sorted(final_allocations.items(), key=lambda x: x[1], reverse=True)
        for i, (etf, allocation) in enumerate(sorted_allocations[:5], 1):
            print(f"{i}. {etf}: {allocation:.1f}%")
        
        print("\n✓ Complete allocation pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def test_individual_agents():
    """Test individual agents with sample data."""
    
    print("\nIndividual Agents Test")
    print("="*30)
    
    try:
        # Create sample data
        etf_data, macro_data, debate_output, analyst_scores = create_allocation_sample_data()
        universe = ['SPY', 'QQQ', 'TLT', 'GLD']
        
        # Test Trader Agent
        print("\n1. Testing Trader Agent...")
        trader = TraderAgent("TestTrader")
        
        trader_data = {
            'debate_output': debate_output,
            'analyst_scores': analyst_scores,
            'universe': universe,
            'timestamp': datetime.now().isoformat()
        }
        
        trader_result = trader.analyze(trader_data)
        print(f"   Trader analysis completed")
        print(f"   Allocations: {trader_result.get('proposed_allocations', {})}")
        
        # Test Risk Manager Agent
        print("\n2. Testing Risk Manager Agent...")
        risk_manager = RiskManagerAgent("TestRiskManager")
        
        risk_data = {
            'proposed_allocations': trader_result.get('proposed_allocations', {}),
            'macro_data': macro_data,
            'risk_factors': {
                'volatility_regime': 'high',
                'geopolitical_risk': 'elevated'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        risk_result = risk_manager.analyze(risk_data)
        print(f"   Risk manager analysis completed")
        print(f"   Allocations: {risk_result.get('risk_adjusted_allocations', {})}")
        
        # Test Portfolio Optimizer Agent
        print("\n3. Testing Portfolio Optimizer Agent...")
        optimizer = PortfolioOptimizerAgent("TestOptimizer")
        
        optimizer_data = {
            'risk_adjusted_allocations': risk_result.get('risk_adjusted_allocations', {}),
            'etf_data': etf_data,
            'universe': universe,
            'timestamp': datetime.now().isoformat()
        }
        
        optimizer_result = optimizer.analyze(optimizer_data)
        print(f"   Portfolio optimizer analysis completed")
        print(f"   Allocations: {optimizer_result.get('final_allocations', {})}")
        
        print("\n✓ All individual agent tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Individual agents test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Allocation Agents Test Suite")
    print("="*50)
    
    # Test complete pipeline
    test_complete_allocation_pipeline()
    
    # Test individual agents
    test_individual_agents()
    
    print("\n" + "="*50)
    print("All tests completed!")

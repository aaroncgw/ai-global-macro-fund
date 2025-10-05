"""
Test Revamped Allocation Agents Integration

This module tests the complete allocation pipeline from risk manager
to portfolio agent for ETF allocation decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.agents.risk_manager import RiskManager
from src.agents.portfolio_manager import PortfolioManagerAgent


def create_allocation_sample_data():
    """Create sample data for testing the allocation pipeline."""
    
    # Create sample ETF data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    etf_data = pd.DataFrame({
        'SPY': 100 + np.cumsum(np.random.randn(252) * 0.01),
        'TLT': 150 + np.cumsum(np.random.randn(252) * 0.005),
        'GLD': 180 + np.cumsum(np.random.randn(252) * 0.008),
        'EWJ': 120 + np.cumsum(np.random.randn(252) * 0.012),
        'FXI': 80 + np.cumsum(np.random.randn(252) * 0.018)
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
    
    # Create sample macro and geo scores
    macro_scores = {
        'SPY': {'score': 0.4, 'confidence': 0.8, 'reason': 'Strong economic growth supports equity markets'},
        'TLT': {'score': -0.8, 'confidence': 0.9, 'reason': 'Rising rates create headwinds for bonds'},
        'GLD': {'score': 0.7, 'confidence': 0.7, 'reason': 'Inflation hedge demand supports gold'},
        'EWJ': {'score': -0.3, 'confidence': 0.6, 'reason': 'Mixed signals from Japanese economy'},
        'FXI': {'score': -0.5, 'confidence': 0.8, 'reason': 'Chinese economic headwinds persist'}
    }
    
    geo_scores = {
        'SPY': {'score': -0.4, 'confidence': 0.7, 'reason': 'Geopolitical tensions affect US markets'},
        'TLT': {'score': 0.3, 'confidence': 0.6, 'reason': 'Safe haven demand during uncertainty'},
        'GLD': {'score': 0.7, 'confidence': 0.8, 'reason': 'Geopolitical risks drive gold demand'},
        'EWJ': {'score': 0.5, 'confidence': 0.7, 'reason': 'Regional stability supports Japanese assets'},
        'FXI': {'score': -0.8, 'confidence': 0.9, 'reason': 'Trade tensions impact Chinese markets'}
    }
    
    return etf_data, macro_data, macro_scores, geo_scores


def test_complete_allocation_pipeline():
    """Test the complete allocation pipeline from risk manager to portfolio agent."""
    
    print("Complete Revamped Allocation Pipeline Test")
    print("="*50)
    
    try:
        # Create sample data
        etf_data, macro_data, macro_scores, geo_scores = create_allocation_sample_data()
        universe = ['SPY', 'TLT', 'GLD', 'EWJ', 'FXI']
        
        print(f"✓ Sample data created")
        print(f"  ETFs: {len(universe)}")
        print(f"  Macro indicators: {len(macro_data)}")
        print(f"  Macro scores: {len(macro_scores)}")
        print(f"  Geo scores: {len(geo_scores)}")
        
        # Initialize all agents
        risk_manager = RiskManager("RiskManager")
        portfolio_manager = PortfolioManagerAgent("PortfolioManagerAgent")
        
        print(f"✓ All agents initialized")
        
        # Create initial state
        state = {
            'macro_scores': macro_scores,
            'geo_scores': geo_scores,
            'macro_data': macro_data,
            'etf_data': etf_data,
            'universe': universe,
            'agent_reasoning': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Risk Manager combines scores and adjusts for risk
        print("\n1. Running Risk Manager...")
        state = risk_manager.assess(state)
        risk_assessments = state.get('risk_assessments', {})
        print(f"   Risk assessments generated for {len(risk_assessments)} ETFs")
        
        # Step 2: Portfolio Agent optimizes allocations
        print("\n2. Running Portfolio Agent...")
        state = portfolio_manager.manage(state)
        final_allocations = state.get('final_allocations', {})
        print(f"   Final allocations generated for {len(final_allocations)} ETFs")
        
        # Display results
        print("\n" + "="*50)
        print("ALLOCATION RESULTS")
        print("="*50)
        
        # Show risk assessments
        print("\nRisk Assessments:")
        print("="*20)
        
        for etf, assessment in risk_assessments.items():
            if isinstance(assessment, dict):
                risk_level = assessment.get('risk_level', 'unknown')
                adjusted_score = assessment.get('adjusted_score', 0.0)
                reason = assessment.get('reason', 'No reasoning provided')
                print(f"{etf}: {risk_level.upper()} Risk (Score: {adjusted_score:.3f})")
                print(f"  Reason: {reason[:60]}...")
                print()
        
        # Show final allocations
        print("Final Portfolio Allocations:")
        print("="*30)
        
        for etf, allocation in final_allocations.items():
            if isinstance(allocation, dict):
                action = allocation.get('action', 'unknown')
                allocation_pct = allocation.get('allocation', 0.0)
                reason = allocation.get('reason', 'No reasoning provided')
                print(f"{etf}: {action.upper()} {allocation_pct:.1%}")
                print(f"  Reason: {reason[:60]}...")
                print()
            else:
                print(f"{etf}: {allocation}")
        
        # Show top allocations
        print("Top Allocations:")
        print("="*15)
        
        sorted_allocations = []
        for etf, allocation in final_allocations.items():
            if isinstance(allocation, dict):
                allocation_pct = allocation.get('allocation', 0.0)
                sorted_allocations.append((etf, allocation_pct))
            else:
                sorted_allocations.append((etf, float(allocation)))
        
        sorted_allocations.sort(key=lambda x: x[1], reverse=True)
        for i, (etf, allocation) in enumerate(sorted_allocations[:5], 1):
            print(f"{i}. {etf}: {allocation:.1%}")
        
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
        etf_data, macro_data, macro_scores, geo_scores = create_allocation_sample_data()
        universe = ['SPY', 'TLT', 'GLD']
        
        # Test Risk Manager Agent
        print("\n1. Testing Risk Manager Agent...")
        risk_manager = RiskManager("TestRiskManager")
        
        risk_data = {
            'macro_scores': macro_scores,
            'geo_scores': geo_scores,
            'macro_data': macro_data,
            'etf_data': etf_data,
            'universe': universe,
            'agent_reasoning': {},
            'timestamp': datetime.now().isoformat()
        }
        
        risk_result = risk_manager.assess(risk_data)
        print(f"   Risk manager analysis completed")
        risk_assessments = risk_result.get('risk_assessments', {})
        print(f"   Risk assessments: {len(risk_assessments)} ETFs")
        
        # Test Portfolio Agent
        print("\n2. Testing Portfolio Agent...")
        portfolio_manager = PortfolioManagerAgent("TestPortfolioManagerAgent")
        
        portfolio_data = {
            'risk_assessments': risk_assessments,
            'etf_data': etf_data,
            'universe': universe,
            'agent_reasoning': {},
            'timestamp': datetime.now().isoformat()
        }
        
        portfolio_result = portfolio_manager.manage(portfolio_data)
        print(f"   Portfolio agent analysis completed")
        final_allocations = portfolio_result.get('final_allocations', {})
        print(f"   Final allocations: {len(final_allocations)} ETFs")
        
        # Show sample results
        print("\nSample Results:")
        print("="*15)
        
        for etf in universe:
            if etf in risk_assessments:
                risk_assessment = risk_assessments[etf]
                if isinstance(risk_assessment, dict):
                    risk_level = risk_assessment.get('risk_level', 'unknown')
                    adjusted_score = risk_assessment.get('adjusted_score', 0.0)
                    print(f"  {etf}: {risk_level.upper()} Risk (Score: {adjusted_score:.3f})")
            
            if etf in final_allocations:
                allocation = final_allocations[etf]
                if isinstance(allocation, dict):
                    action = allocation.get('action', 'unknown')
                    allocation_pct = allocation.get('allocation', 0.0)
                    print(f"  {etf}: {action.upper()} {allocation_pct:.1%}")
        
        print("\n✓ All individual agent tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Individual agents test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Revamped Allocation Agents Test Suite")
    print("="*50)
    
    # Test complete pipeline
    test_complete_allocation_pipeline()
    
    # Test individual agents
    test_individual_agents()
    
    print("\n" + "="*50)
    print("All tests completed!")

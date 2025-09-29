"""
Test Complete Macro Trading Workflow

This module tests the complete LangGraph workflow from data fetching
to final portfolio optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.graph.macro_trading_graph import MacroTradingGraph


def test_complete_workflow():
    """Test the complete macro trading workflow."""
    
    print("Complete Macro Trading Workflow Test")
    print("="*50)
    
    try:
        # Initialize the graph
        graph = MacroTradingGraph(debug=True)
        print("✓ Macro trading graph initialized")
        
        # Test with different ETF universes
        test_cases = [
            {
                'name': 'US Focused Portfolio',
                'universe': ['SPY', 'QQQ', 'TLT', 'GLD', 'UUP'],
                'description': 'US-focused portfolio with bonds, gold, and dollar'
            },
            {
                'name': 'Global Diversified Portfolio',
                'universe': ['SPY', 'EWJ', 'EWG', 'FXI', 'GLD', 'TLT'],
                'description': 'Globally diversified portfolio across regions'
            },
            {
                'name': 'Commodity Focused Portfolio',
                'universe': ['GLD', 'SLV', 'USO', 'UNG', 'DBC'],
                'description': 'Commodity-focused portfolio for inflation hedging'
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing {test_case['name']}")
            print(f"   Description: {test_case['description']}")
            print(f"   Universe: {', '.join(test_case['universe'])}")
            
            # Run the workflow
            start_time = datetime.now()
            final_allocations = graph.propagate(test_case['universe'])
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            print(f"   ✓ Workflow completed in {duration:.1f} seconds")
            print(f"   Final allocations: {final_allocations}")
            
            # Store results
            results[test_case['name']] = {
                'universe': test_case['universe'],
                'allocations': final_allocations,
                'duration': duration
            }
            
            # Show allocation summary
            if final_allocations:
                print(f"   Allocation Summary:")
                sorted_allocations = sorted(final_allocations.items(), key=lambda x: x[1], reverse=True)
                for j, (etf, allocation) in enumerate(sorted_allocations, 1):
                    print(f"     {j}. {etf}: {allocation:.1f}%")
        
        # Summary of all results
        print("\n" + "="*50)
        print("WORKFLOW RESULTS SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Universe: {', '.join(result['universe'])}")
            print(f"  Duration: {result['duration']:.1f} seconds")
            print(f"  Top Allocation: {max(result['allocations'].items(), key=lambda x: x[1])}")
        
        print("\n✓ Complete workflow test completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()


def test_workflow_components():
    """Test individual workflow components."""
    
    print("\nWorkflow Components Test")
    print("="*30)
    
    try:
        # Initialize the graph
        graph = MacroTradingGraph(debug=False)
        print("✓ Graph initialized for component testing")
        
        # Test with small universe for faster execution
        test_universe = ['SPY', 'QQQ', 'TLT']
        print(f"✓ Testing with universe: {', '.join(test_universe)}")
        
        # Run workflow
        final_allocations = graph.propagate(test_universe)
        print(f"✓ Component test completed")
        print(f"  Final allocations: {final_allocations}")
        
        # Verify allocations sum to 100%
        total_allocation = sum(final_allocations.values())
        print(f"  Total allocation: {total_allocation:.1f}%")
        
        if abs(total_allocation - 100.0) < 0.1:
            print("  ✓ Allocations properly normalized")
        else:
            print("  ⚠ Allocations may not be properly normalized")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling in the workflow."""
    
    print("\nError Handling Test")
    print("="*20)
    
    try:
        # Initialize the graph
        graph = MacroTradingGraph(debug=False)
        print("✓ Graph initialized for error testing")
        
        # Test with empty universe
        print("1. Testing empty universe...")
        empty_allocations = graph.propagate([])
        print(f"   Empty universe result: {empty_allocations}")
        
        # Test with invalid ETF symbols
        print("2. Testing invalid ETF symbols...")
        invalid_allocations = graph.propagate(['INVALID1', 'INVALID2'])
        print(f"   Invalid symbols result: {invalid_allocations}")
        
        # Test with single ETF
        print("3. Testing single ETF...")
        single_allocations = graph.propagate(['SPY'])
        print(f"   Single ETF result: {single_allocations}")
        
        print("✓ Error handling test completed")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Macro Trading Workflow Test Suite")
    print("="*50)
    
    # Test complete workflow
    test_complete_workflow()
    
    # Test workflow components
    test_workflow_components()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "="*50)
    print("All workflow tests completed!")

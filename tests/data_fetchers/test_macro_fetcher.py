"""
Test script for MacroFetcher functionality.

This script tests the macro data fetcher with sample data to ensure
all components are working correctly.
"""

import os
import sys
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_fetchers.macro_fetcher import MacroFetcher


def test_macro_fetcher():
    """Test the MacroFetcher with sample data."""
    print("Testing MacroFetcher...")
    
    try:
        # Initialize fetcher
        fetcher = MacroFetcher()
        print("✓ MacroFetcher initialized successfully")
        
        # Test with sample ETFs
        sample_etfs = ['SPY', 'QQQ', 'TLT', 'GLD']
        print(f"Testing with ETFs: {sample_etfs}")
        
        # Test ETF data fetching
        print("\n1. Testing ETF data fetching...")
        etf_data = fetcher.fetch_etf_data(sample_etfs, start='2023-01-01')
        if not etf_data.empty:
            print(f"✓ ETF data fetched successfully: {etf_data.shape}")
        else:
            print("⚠ No ETF data retrieved")
        
        # Test ETF returns calculation
        print("\n2. Testing ETF returns calculation...")
        returns_data = fetcher.fetch_etf_returns(sample_etfs, start='2023-01-01')
        if not returns_data.empty:
            print(f"✓ ETF returns calculated successfully: {returns_data.shape}")
        else:
            print("⚠ No returns data calculated")
        
        # Test macro indicators (if FRED API key is available)
        print("\n3. Testing macro indicators...")
        sample_indicators = ['CPIAUCSL', 'UNRATE']
        macro_data = fetcher.fetch_macro_data(sample_indicators)
        if macro_data:
            print(f"✓ Macro data fetched for {len(macro_data)} indicators")
            for indicator, data in macro_data.items():
                if 'error' not in data:
                    print(f"  - {indicator}: {data.get('periods', 0)} periods")
                else:
                    print(f"  - {indicator}: Error - {data.get('error', 'Unknown error')}")
        else:
            print("⚠ No macro data retrieved (check FRED_API_KEY)")
        
        # Test news fetching (if Finlight API key is available)
        print("\n4. Testing news fetching...")
        news_data = fetcher.fetch_geopolitical_news('macro economic indicators')
        if news_data:
            print(f"✓ News data fetched: {len(news_data)} articles")
        else:
            print("⚠ No news data retrieved (check FINLIGHT_API_KEY)")
        
        # Test comprehensive data fetch
        print("\n5. Testing comprehensive data fetch...")
        comprehensive_data = fetcher.fetch_comprehensive_data(
            etfs=sample_etfs,
            indicators=sample_indicators,
            news_queries=['macro economic indicators']
        )
        
        summary = fetcher.get_data_summary(comprehensive_data)
        print("✓ Comprehensive data fetch completed")
        print("Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_configuration_integration():
    """Test integration with the configuration system."""
    print("\n" + "="*50)
    print("Testing configuration integration...")
    
    try:
        from config import ETF_UNIVERSE, MACRO_INDICATORS
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - ETF Universe: {len(ETF_UNIVERSE)} ETFs")
        print(f"  - Macro Indicators: {len(MACRO_INDICATORS)} indicators")
        
        # Test with configuration data
        fetcher = MacroFetcher()
        
        # Test with a subset of the configured ETFs
        test_etfs = ETF_UNIVERSE[:5]  # First 5 ETFs
        test_indicators = MACRO_INDICATORS[:2]  # First 2 indicators
        
        print(f"\nTesting with configured data:")
        print(f"  - ETFs: {test_etfs}")
        print(f"  - Indicators: {test_indicators}")
        
        # Fetch data
        data = fetcher.fetch_comprehensive_data(
            etfs=test_etfs,
            indicators=test_indicators,
            news_queries=['global macro trends']
        )
        
        summary = fetcher.get_data_summary(data)
        print(f"✓ Configuration integration test completed")
        print(f"  - ETFs processed: {summary['etfs_count']}")
        print(f"  - Indicators processed: {summary['indicators_count']}")
        
    except ImportError as e:
        print(f"⚠ Configuration integration test skipped: {e}")
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")


if __name__ == "__main__":
    print("MacroFetcher Test Suite")
    print("="*50)
    
    # Test basic functionality
    test_macro_fetcher()
    
    # Test configuration integration
    test_configuration_integration()
    
    print("\n" + "="*50)
    print("Test suite completed!")

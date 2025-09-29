"""
Pytest configuration and shared fixtures for the Global Macro ETF Trading System.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add app directory to Python path for web app tests
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))

@pytest.fixture
def sample_etf_universe():
    """Sample ETF universe for testing."""
    return ['SPY', 'QQQ', 'TLT', 'GLD', 'UUP']

@pytest.fixture
def sample_macro_indicators():
    """Sample macro indicators for testing."""
    return ['CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'GDPC1']

@pytest.fixture
def sample_state():
    """Sample state dictionary for testing."""
    return {
        'universe': ['SPY', 'QQQ', 'TLT'],
        'macro_data': {
            'CPIAUCSL': {'latest_value': 300.0, 'periods': 12},
            'UNRATE': {'latest_value': 3.5, 'periods': 12}
        },
        'etf_data': None,  # Will be populated by specific tests
        'news': [
            {'title': 'Test News', 'content': 'Test content', 'sentiment': 'positive'}
        ],
        'analyst_scores': {},
        'agent_reasoning': {},
        'debate_output': [],
        'proposed_allocations': {},
        'risk_adjusted_allocations': {},
        'final_allocations': {}
    }

@pytest.fixture
def mock_etf_data():
    """Mock ETF data for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible tests
    
    data = {}
    for etf in ['SPY', 'QQQ', 'TLT', 'GLD']:
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        data[etf] = {
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }
    
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_news_data():
    """Mock news data for testing."""
    return [
        {
            'title': 'Federal Reserve Raises Interest Rates',
            'content': 'The Federal Reserve announced a 0.25% increase in interest rates...',
            'sentiment': 'negative',
            'published_date': '2024-01-15',
            'url': 'https://example.com/news1'
        },
        {
            'title': 'Strong Economic Growth Reported',
            'content': 'GDP growth exceeded expectations in the latest quarter...',
            'sentiment': 'positive',
            'published_date': '2024-01-14',
            'url': 'https://example.com/news2'
        }
    ]

@pytest.fixture
def mock_macro_data():
    """Mock macro data for testing."""
    return {
        'CPIAUCSL': {
            'latest_value': 300.0,
            'periods': 12,
            'data': {f'2024-{i:02d}-01': 300.0 + i for i in range(1, 13)}
        },
        'UNRATE': {
            'latest_value': 3.5,
            'periods': 12,
            'data': {f'2024-{i:02d}-01': 3.5 + (i * 0.1) for i in range(1, 13)}
        },
        'FEDFUNDS': {
            'latest_value': 5.25,
            'periods': 12,
            'data': {f'2024-{i:02d}-01': 5.25 for i in range(1, 13)}
        },
        'GDPC1': {
            'latest_value': 2.1,
            'periods': 4,
            'data': {f'2024-Q{i}': 2.1 + (i * 0.1) for i in range(1, 5)}
        }
    }

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'debug': True,
        'test_mode': True,
        'mock_llm': True,
        'mock_data': True
    }

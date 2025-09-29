# API Documentation

## 📚 Table of Contents

- [Core Classes](#core-classes)
- [Agent API](#agent-api)
- [Data Fetcher API](#data-fetcher-api)
- [Graph API](#graph-api)
- [Configuration API](#configuration-api)
- [Utility Functions](#utility-functions)

## Core Classes

### BaseAgent

Base class for all AI agents in the system.

```python
class BaseAgent:
    def __init__(self, agent_name: str)
    def analyze(self, data: dict) -> dict
    def llm(self, prompt: str) -> str
    def get_provider_info(self) -> dict
    def switch_provider(self, new_config: dict) -> None
```

**Parameters:**
- `agent_name` (str): Name identifier for the agent

**Methods:**
- `analyze(data: dict) -> dict`: Abstract method for agent analysis
- `llm(prompt: str) -> str`: LLM interface with provider abstraction
- `get_provider_info() -> dict`: Returns current LLM provider information
- `switch_provider(new_config: dict) -> None`: Switch to different LLM provider

### MacroTradingGraph

Main workflow orchestration class using LangGraph.

```python
class MacroTradingGraph:
    def __init__(self, debug: bool = False)
    def propagate(self, universe: list[str], date: str = 'today') -> dict
```

**Parameters:**
- `debug` (bool): Enable debug logging

**Methods:**
- `propagate(universe: list[str], date: str) -> dict`: Run complete workflow

**Returns:**
- Complete state dictionary with final allocations and reasoning

## Agent API

### MacroEconomistAgent

Analyzes macroeconomic indicators and scores ETFs.

```python
class MacroEconomistAgent(BaseAgent):
    def __init__(self, agent_name: str = "MacroEconomistAgent")
    def analyze(self, state: dict) -> dict
```

**Input State:**
```python
{
    'macro_data': {
        'CPIAUCSL': {'latest_value': 300.0, 'periods': 12},
        'UNRATE': {'latest_value': 3.5, 'periods': 12}
    },
    'etf_data': pd.DataFrame(...),
    'universe': ['SPY', 'QQQ', 'TLT']
}
```

**Output State:**
```python
{
    'analyst_scores': {
        'macro': {'SPY': 0.2, 'QQQ': 0.1, 'TLT': -0.3}
    },
    'agent_reasoning': {
        'macro_economist': {
            'scores': {'SPY': 0.2, 'QQQ': 0.1, 'TLT': -0.3},
            'reasoning': 'Macro analysis based on...',
            'key_factors': ['inflation', 'rates'],
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

### GeopoliticalAnalystAgent

Analyzes geopolitical news and events for ETF scoring.

```python
class GeopoliticalAnalystAgent(BaseAgent):
    def __init__(self, agent_name: str = "GeopoliticalAnalystAgent")
    def analyze(self, state: dict) -> dict
```

**Input State:**
```python
{
    'news': [
        {'title': 'Geopolitical Event', 'content': '...'},
        {'title': 'Trade News', 'content': '...'}
    ],
    'universe': ['SPY', 'QQQ', 'TLT']
}
```

**Output State:**
```python
{
    'analyst_scores': {
        'geo': {'SPY': 0.1, 'QQQ': 0.2, 'TLT': 0.0}
    },
    'agent_reasoning': {
        'geopolitical_analyst': {
            'scores': {'SPY': 0.1, 'QQQ': 0.2, 'TLT': 0.0},
            'reasoning': 'Geopolitical analysis based on...',
            'key_factors': ['trade_relations', 'regional_risks'],
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

### CorrelationSpecialistAgent

Analyzes ETF correlations for diversification scoring.

```python
class CorrelationSpecialistAgent(BaseAgent):
    def __init__(self, agent_name: str = "CorrelationSpecialistAgent")
    def analyze(self, state: dict) -> dict
```

**Input State:**
```python
{
    'etf_data': pd.DataFrame(...),  # Price data for correlation calculation
    'universe': ['SPY', 'QQQ', 'TLT']
}
```

**Output State:**
```python
{
    'analyst_scores': {
        'correlation': {'SPY': 0.0, 'QQQ': 0.1, 'TLT': 0.2}
    },
    'agent_reasoning': {
        'correlation_specialist': {
            'scores': {'SPY': 0.0, 'QQQ': 0.1, 'TLT': 0.2},
            'reasoning': 'Correlation analysis based on...',
            'key_factors': ['diversification_benefits', 'portfolio_balance'],
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

### TraderAgent

Proposes initial ETF allocations based on analysis.

```python
class TraderAgent(BaseAgent):
    def __init__(self, agent_name: str = "TraderAgent")
    def analyze(self, data: dict) -> dict
    def propose(self, state: dict) -> dict
```

**Input State:**
```python
{
    'debate_output': [...],
    'analyst_scores': {
        'macro': {...},
        'geo': {...},
        'correlation': {...}
    },
    'universe': ['SPY', 'QQQ', 'TLT']
}
```

**Output State:**
```python
{
    'proposed_allocations': {'SPY': 35.0, 'QQQ': 40.0, 'TLT': 25.0},
    'agent_reasoning': {
        'trader': {
            'proposed_allocations': {'SPY': 35.0, 'QQQ': 40.0, 'TLT': 25.0},
            'reasoning': 'Trader proposal based on...',
            'key_factors': ['macro', 'geo', 'correlation'],
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

### RiskManagerAgent

Assesses and adjusts allocations for risk factors.

```python
class RiskManagerAgent(BaseAgent):
    def __init__(self, agent_name: str = "RiskManagerAgent")
    def analyze(self, data: dict) -> dict
    def assess(self, state: dict) -> dict
```

**Input State:**
```python
{
    'proposed_allocations': {'SPY': 35.0, 'QQQ': 40.0, 'TLT': 25.0},
    'macro_data': {...},
    'risk_factors': {...}
}
}
```

**Output State:**
```python
{
    'risk_adjusted_allocations': {'SPY': 25.0, 'QQQ': 30.0, 'TLT': 45.0},
    'agent_reasoning': {
        'risk_manager': {
            'risk_adjusted_allocations': {'SPY': 25.0, 'QQQ': 30.0, 'TLT': 45.0},
            'reasoning': 'Risk assessment based on...',
            'risk_factors': ['volatility', 'correlation'],
            'adjustments': {'SPY': -10.0, 'QQQ': -10.0, 'TLT': 20.0},
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

### PortfolioOptimizerAgent

Performs mathematical optimization for final allocations.

```python
class PortfolioOptimizerAgent:
    def __init__(self, agent_name: str = "PortfolioOptimizerAgent")
    def analyze(self, data: dict) -> dict
    def optimize(self, state: dict) -> dict
```

**Input State:**
```python
{
    'risk_adjusted_allocations': {'SPY': 25.0, 'QQQ': 30.0, 'TLT': 45.0},
    'etf_data': pd.DataFrame(...),
    'universe': ['SPY', 'QQQ', 'TLT']
}
```

**Output State:**
```python
{
    'final_allocations': {'SPY': 25.0, 'QQQ': 25.0, 'TLT': 50.0},
    'agent_reasoning': {
        'portfolio_optimizer': {
            'final_allocations': {'SPY': 25.0, 'QQQ': 25.0, 'TLT': 50.0},
            'reasoning': 'Portfolio optimization using...',
            'optimization_method': 'mean_variance',
            'constraints': {'sum_to_one': True, 'non_negative': True},
            'performance_metrics': {'expected_return': 'calculated', 'volatility': 'calculated'},
            'timestamp': '2024-01-01T12:00:00'
        }
    }
}
```

## Data Fetcher API

### MacroFetcher

Unified data fetching for all data sources.

```python
class MacroFetcher:
    def __init__(self)
    def fetch_macro_data(self, indicators: list[str]) -> dict
    def fetch_etf_data(self, etfs: list[str], start_date: str = '2020-01-01', end_date: str = 'today') -> pd.DataFrame
    def fetch_etf_returns(self, etf_data: pd.DataFrame) -> pd.DataFrame
    def fetch_geopolitical_news(self, query: str = 'geopolitical events macro impact', limit: int = 20) -> list[dict]
    def fetch_comprehensive_data(self, etfs: list[str], indicators: list[str], start_date: str = '2020-01-01', end_date: str = 'today', news_queries: list[str] = ['global macro trends']) -> dict
```

**Methods:**

#### fetch_macro_data(indicators: list[str]) -> dict
Fetches macroeconomic data from FRED API.

**Parameters:**
- `indicators` (list[str]): List of FRED indicator codes

**Returns:**
```python
{
    'CPIAUCSL': {'latest_value': 300.0, 'periods': 12, 'data': {...}},
    'UNRATE': {'latest_value': 3.5, 'periods': 12, 'data': {...}}
}
```

#### fetch_etf_data(etfs: list[str], start_date: str, end_date: str) -> pd.DataFrame
Fetches ETF price data from yfinance.

**Parameters:**
- `etfs` (list[str]): List of ETF symbols
- `start_date` (str): Start date for data
- `end_date` (str): End date for data

**Returns:**
```python
pd.DataFrame({
    'SPY': {'Open': [...], 'High': [...], 'Low': [...], 'Close': [...], 'Adj Close': [...]},
    'QQQ': {'Open': [...], 'High': [...], 'Low': [...], 'Close': [...], 'Adj Close': [...]}
})
```

#### fetch_geopolitical_news(query: str, limit: int) -> list[dict]
Fetches geopolitical news from Finlight.me API.

**Parameters:**
- `query` (str): Search query for news
- `limit` (int): Maximum number of articles

**Returns:**
```python
[
    {
        'title': 'Geopolitical Event',
        'content': 'Event description...',
        'url': 'https://...',
        'published_date': '2024-01-01',
        'sentiment': 'positive'
    }
]
```

#### fetch_comprehensive_data(etfs: list[str], indicators: list[str], start_date: str, end_date: str, news_queries: list[str]) -> dict
Fetches all required data in a single call.

**Parameters:**
- `etfs` (list[str]): List of ETF symbols
- `indicators` (list[str]): List of FRED indicators
- `start_date` (str): Start date for ETF data
- `end_date` (str): End date for ETF data
- `news_queries` (list[str]): List of news search queries

**Returns:**
```python
{
    'fetch_timestamp': '2024-01-01T12:00:00',
    'etfs_count': 4,
    'indicators_count': 4,
    'macro_indicators': {...},
    'etf_data': pd.DataFrame(...),
    'etf_returns': pd.DataFrame(...),
    'news_articles': [...],
    'errors': []
}
```

## Graph API

### MacroTradingGraph

Main workflow orchestration class.

```python
class MacroTradingGraph:
    def __init__(self, debug: bool = False)
    def propagate(self, universe: list[str], date: str = 'today') -> dict
```

#### propagate(universe: list[str], date: str) -> dict
Runs the complete macro trading workflow.

**Parameters:**
- `universe` (list[str]): List of ETF symbols to analyze
- `date` (str): Date for analysis (default: 'today')

**Returns:**
```python
{
    'universe': ['SPY', 'QQQ', 'TLT'],
    'final_allocations': {'SPY': 25.0, 'QQQ': 25.0, 'TLT': 50.0},
    'agent_reasoning': {
        'macro_economist': {...},
        'geopolitical_analyst': {...},
        'correlation_specialist': {...},
        'debate': {...},
        'trader': {...},
        'risk_manager': {...},
        'portfolio_optimizer': {...}
    },
    'analyst_scores': {
        'macro': {...},
        'geo': {...},
        'correlation': {...}
    },
    'debate_output': [...],
    'proposed_allocations': {...},
    'risk_adjusted_allocations': {...}
}
```

## Configuration API

### System Configuration

```python
# src/config.py

# ETF Universe
ETF_UNIVERSE = [
    'SPY', 'QQQ', 'TLT', 'GLD',  # Add more ETFs
]

# Macro Indicators
MACRO_INDICATORS = ['CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'GDPC1']

# Default Configuration
DEFAULT_CONFIG = {
    'max_debate_rounds': 2,
    'llm_model': 'deepseek-chat',
    'analysis_horizon': 'long_term',
    'rebalance_frequency': 'monthly',
    'risk_tolerance': 'moderate'
}

# LLM Configuration
LLM_CONFIG = {
    'provider': 'deepseek',
    'model': 'deepseek-chat',
    'api_key_env': 'DEEPSEEK_API_KEY',
    'base_url': 'https://api.deepseek.com/v1'
}
```

### Environment Variables

```bash
# Required
FRED_API_KEY=your_fred_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Optional
FINLIGHT_API_KEY=your_finlight_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Utility Functions

### Display Functions

```python
def display_comprehensive_reasoning(complete_state: dict, allocations: dict) -> None
def generate_allocation_rationale(complete_state: dict, allocations: dict) -> str
def generate_key_insights(complete_state: dict, allocations: dict) -> list[str]
```

### Validation Functions

```python
def validate_config() -> None
def validate_etf_universe(universe: list[str]) -> list[str]
def validate_date(date_str: str) -> bool
```

### Performance Functions

```python
def time_function(func_name: str) -> callable
def cleanup_state(state: dict) -> dict
def track_error(error: Exception, context: str) -> None
```

## Error Handling

### Common Exceptions

```python
class AgentError(Exception):
    """Base exception for agent-related errors"""
    pass

class DataFetchError(Exception):
    """Exception for data fetching errors"""
    pass

class LLMError(Exception):
    """Exception for LLM-related errors"""
    pass

class OptimizationError(Exception):
    """Exception for portfolio optimization errors"""
    pass
```

### Error Handling Patterns

```python
try:
    result = agent.analyze(state)
except AgentError as e:
    logger.error(f"Agent error: {e}")
    # Return neutral scores
    state['analyst_scores']['agent'] = {etf: 0.0 for etf in state['universe']}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle gracefully
```

## Performance Considerations

### Caching

```python
@functools.lru_cache(maxsize=128)
def cached_fred_data(api_key: str, indicator: str, periods: int):
    """Cache FRED data for 1 hour"""
    pass
```

### Memory Management

```python
def cleanup_state(state: dict) -> dict:
    """Remove large data structures after processing"""
    if 'etf_data' in state:
        del state['etf_data']
    return state
```

### Parallel Processing

```python
import ray

@ray.remote
def parallel_agent_analysis(agent, state):
    return agent.analyze(state)

# Execute agents in parallel
futures = [parallel_agent_analysis.remote(agent, state) for agent in agents]
results = ray.get(futures)
```

## Testing API

### Unit Testing

```python
import pytest
from src.agents.macro_economist import MacroEconomistAgent

def test_macro_economist():
    agent = MacroEconomistAgent()
    state = {
        'macro_data': {'CPIAUCSL': {'latest_value': 300.0}},
        'universe': ['SPY', 'QQQ']
    }
    result = agent.analyze(state)
    assert 'analyst_scores' in result
```

### Integration Testing

```python
def test_complete_workflow():
    graph = MacroTradingGraph(debug=True)
    result = graph.propagate(['SPY', 'QQQ'], '2024-01-01')
    assert 'final_allocations' in result
    assert len(result['final_allocations']) == 2
```

---

This API documentation provides comprehensive reference for all classes, methods, and data structures in the Global Macro ETF Trading System. For implementation details, refer to the source code and docstrings.

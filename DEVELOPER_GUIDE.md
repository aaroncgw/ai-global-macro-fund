# Developer Guide: Global Macro ETF Trading System

This guide provides comprehensive technical documentation for developers working on the Global Macro ETF Trading System. It covers system architecture, development workflows, and best practices for extending the system.

## 🏗️ System Architecture Deep Dive

### Core Components

```
src/
├── agents/                 # AI Agent implementations
│   ├── base_agent.py      # Base class for all agents
│   ├── macro_economist.py # Macro economic analysis
│   ├── geopolitical_analyst.py # Geopolitical risk analysis
│   ├── risk_manager.py    # Risk assessment
│   └── portfolio_agent.py # Portfolio optimization
├── data_fetchers/         # Data source integrations
│   └── macro_fetcher.py   # Unified data fetching
├── graph/                 # LangGraph workflow orchestration
│   └── macro_trading_graph.py # Main workflow definition
├── config.py             # System configuration
└── main.py              # Entry point
```

### Agent Architecture

All agents inherit from `BaseAgent` which provides:

```python
class BaseAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = self._initialize_llm()
    
    def analyze(self, data: dict) -> dict:
        """Abstract method - must be implemented by subclasses"""
        raise NotImplementedError
    
    def llm(self, prompt: str) -> str:
        """LLM interface with provider abstraction"""
        pass
```

### State Management

The system uses LangGraph's state management for data flow:

```python
state = {
    'universe': ['SPY', 'QQQ', 'TLT'],
    'macro_data': {...},
    'etf_data': pd.DataFrame(...),
    'news': [...],
    'analyst_scores': {
        'macro': {...},
        'geo': {...},
        'correlation': {...}
    },
    'agent_reasoning': {
        'macro_economist': {...},
        'geopolitical_analyst': {...},
        # ... other agents
    },
    'macro_scores': {...},
    'geo_scores': {...},
    'risk_assessments': {...},
    'final_allocations': {...}
}
```

## 🔧 Development Workflow

### 1. Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd ai-hedge-fund

# Install dependencies
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Verify installation
poetry run python src/main.py --universe SPY QQQ --debug
```

### 2. Understanding the Workflow

The system follows a strict LangGraph workflow:

```python
# Workflow Definition (src/graph/macro_trading_graph.py)
self.graph.add_node('fetch', fetch_node)
self.graph.add_node('macro_analyst', macro_analyst_node)
self.graph.add_node('geo_analyst', geo_analyst_node)
self.graph.add_node('risk', risk_node)
self.graph.add_node('portfolio', portfolio_node)

# Define edges
self.graph.set_entry_point('fetch')
self.graph.add_edge('fetch', 'macro_analyst')
self.graph.add_edge('macro_analyst', 'geo_analyst')
# ... etc
```

### 3. Adding New Agents

#### Step 1: Create Agent Class

```python
# src/agents/new_agent.py
from src.agents.base_agent import BaseAgent
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class NewAgent(BaseAgent):
    def __init__(self, agent_name: str = "NewAgent"):
        super().__init__(agent_name)
        self.specialization = "new_analysis"
        self.analysis_focus = "specific_focus"
    
    def analyze(self, state: dict) -> dict:
        """Analyze data and update state."""
        try:
            # Extract data from state
            input_data = state.get('input_data', {})
            universe = state.get('universe', [])
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(input_data, universe)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse results
            scores = self._parse_scores(response, universe)
            
            # Update state
            state['analyst_scores']['new_analysis'] = scores
            
            # Store detailed reasoning
            state['agent_reasoning'] = state.get('agent_reasoning', {})
            state['agent_reasoning']['new_agent'] = {
                'scores': scores,
                'reasoning': f"New agent analysis based on {len(input_data)} data points",
                'key_factors': ['factor1', 'factor2'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"New agent analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"New agent analysis failed: {e}")
            # Return neutral scores on error
            neutral_scores = {etf: 0.0 for etf in state.get('universe', [])}
            state['analyst_scores']['new_analysis'] = neutral_scores
            return state
    
    def _create_analysis_prompt(self, input_data: dict, universe: list) -> str:
        """Create analysis prompt for LLM."""
        return f"""
        As a new analyst, analyze the following data and score each ETF:
        
        INPUT DATA: {input_data}
        UNIVERSE: {', '.join(universe)}
        
        Return ONLY a JSON object with ETF scores:
        {{"SPY": 0.2, "QQQ": 0.1, ...}}
        """
    
    def _parse_scores(self, response: str, universe: list) -> dict:
        """Parse scores from LLM response."""
        # Implementation here
        pass
```

#### Step 2: Integrate with Workflow

```python
# src/graph/macro_trading_graph.py
from src.agents.new_agent import NewAgent

class MacroTradingGraph:
    def __init__(self, debug=False):
        # ... existing code ...
        self.new_agent = NewAgent("NewAgent")
    
    def add_nodes_and_edges(self):
        # ... existing nodes ...
        
        def new_agent_node(state):
            """Run new agent analysis."""
            try:
                logger.info("Running new agent analysis...")
                result = self.new_agent.analyze(state)
                logger.info("New agent analysis completed")
                return state
            except Exception as e:
                logger.error(f"New agent analysis failed: {e}")
                return state
        
        # Add node
        self.graph.add_node('new_agent', new_agent_node)
        
        # Add edges (insert at appropriate position)
        self.graph.add_edge('previous_node', 'new_agent')
        self.graph.add_edge('new_agent', 'next_node')
```

#### Step 3: Update Display Logic

```python
# src/main.py
def display_comprehensive_reasoning(complete_state, allocations):
    # ... existing code ...
    
    # Display new agent reasoning
    if 'new_agent' in agent_reasoning:
        new_data = agent_reasoning['new_agent']
        print("\n🆕 NEW AGENT ANALYSIS:")
        print("-" * 40)
        print(f"Key Factors: {', '.join(new_data.get('key_factors', ['Not specified']))}")
        print(f"Reasoning: {new_data.get('reasoning', 'No detailed reasoning provided')}")
        if new_data.get('scores'):
            print("ETF Scores:")
            for etf, score in new_data['scores'].items():
                print(f"  {etf}: {score:.2f}")
```

### 4. Adding New Data Sources

#### Step 1: Create Data Fetcher

```python
# src/data_fetchers/new_fetcher.py
import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class NewDataFetcher:
    def __init__(self):
        self.api_key = os.getenv('NEW_API_KEY')
        self.base_url = 'https://api.newprovider.com/v1'
    
    def fetch_new_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from new source."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(f"{self.base_url}/data", 
                                 headers=headers, 
                                 params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch new data: {e}")
            return {}
```

#### Step 2: Integrate with MacroFetcher

```python
# src/data_fetchers/macro_fetcher.py
from src.data_fetchers.new_fetcher import NewDataFetcher

class MacroFetcher:
    def __init__(self):
        # ... existing code ...
        self.new_fetcher = NewDataFetcher()
    
    def fetch_comprehensive_data(self, etfs, indicators, **kwargs):
        # ... existing code ...
        
        # Add new data source
        new_data = self.new_fetcher.fetch_new_data(kwargs.get('new_params', {}))
        all_data["new_data"] = new_data
        
        return all_data
```

### 5. Modifying LLM Providers

#### Step 1: Update BaseAgent

```python
# src/agents/base_agent.py
def _initialize_llm(self):
    if LLM_CONFIG['provider'] == 'new_provider':
        from langchain_new_provider import NewProviderLLM
        return NewProviderLLM(
            api_key=os.getenv(LLM_CONFIG['api_key_env']),
            model=LLM_CONFIG['model'],
            base_url=LLM_CONFIG.get('base_url')
        )
    # ... existing providers
```

#### Step 2: Update Configuration

```python
# src/config.py
LLM_CONFIG = {
    'provider': 'new_provider',
    'model': 'new_model_name',
    'api_key_env': 'NEW_PROVIDER_API_KEY',
    'base_url': 'https://api.newprovider.com/v1'
}
```

## 🧪 Testing Framework

### Unit Testing

```python
# tests/test_new_agent.py
import pytest
from src.agents.new_agent import NewAgent

class TestNewAgent:
    def test_agent_initialization(self):
        agent = NewAgent("TestAgent")
        assert agent.agent_name == "TestAgent"
        assert agent.specialization == "new_analysis"
    
    def test_analyze_method(self):
        agent = NewAgent("TestAgent")
        state = {
            'universe': ['SPY', 'QQQ'],
            'input_data': {'test': 'data'},
            'analyst_scores': {}
        }
        result = agent.analyze(state)
        assert 'analyst_scores' in result
        assert 'new_analysis' in result['analyst_scores']
```

### Integration Testing

```python
# tests/test_workflow_integration.py
import pytest
from src.graph.macro_trading_graph import MacroTradingGraph

class TestWorkflowIntegration:
    def test_complete_workflow(self):
        graph = MacroTradingGraph(debug=True)
        result = graph.propagate(['SPY', 'QQQ'], '2024-01-01')
        
        assert 'final_allocations' in result
        assert 'agent_reasoning' in result
        assert len(result['final_allocations']) == 2
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_new_agent.py

# Run with coverage
poetry run pytest --cov=src

# Run integration tests
poetry run pytest tests/integration/
```

## 🔍 Debugging and Troubleshooting

### Debug Mode

Enable comprehensive logging:

```python
# src/main.py
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")
```

### Common Debug Scenarios

1. **Agent Not Responding**:
   ```python
   # Check LLM connectivity
   agent = MacroEconomistAgent()
   response = agent.llm("Test prompt")
   print(f"LLM Response: {response}")
   ```

2. **Data Fetching Issues**:
   ```python
   # Test individual data sources
   fetcher = MacroFetcher()
   macro_data = fetcher.fetch_macro_data(['CPIAUCSL'])
   print(f"Macro Data: {macro_data}")
   ```

3. **State Management Issues**:
   ```python
   # Inspect state at each step
   def debug_node(state):
       print(f"State keys: {list(state.keys())}")
       print(f"Analyst scores: {state.get('analyst_scores', {})}")
       return state
   ```

### Performance Profiling

```python
# Add timing to workflow
import time

def timed_node(node_func):
    def wrapper(state):
        start_time = time.time()
        result = node_func(state)
        end_time = time.time()
        logger.info(f"Node {node_func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 📊 Data Flow Analysis

### Understanding State Transitions

```python
# Initial State
initial_state = {
    'universe': ['SPY', 'QQQ', 'TLT'],
    'timestamp': '2024-01-01'
}

# After Data Fetching
state_after_fetch = {
    'universe': ['SPY', 'QQQ', 'TLT'],
    'macro_data': {...},
    'etf_data': pd.DataFrame(...),
    'news': [...],
    'analyst_scores': {},
    'agent_reasoning': {}
}

# After Analysis Phase
state_after_analysis = {
    'universe': ['SPY', 'QQQ', 'TLT'],
    'macro_data': {...},
    'etf_data': pd.DataFrame(...),
    'news': [...],
    'analyst_scores': {
        'macro': {'SPY': 0.2, 'QQQ': 0.1, 'TLT': -0.3},
        'geo': {'SPY': 0.1, 'QQQ': 0.2, 'TLT': 0.0},
        'correlation': {'SPY': 0.0, 'QQQ': 0.1, 'TLT': 0.2}
    },
    'agent_reasoning': {...}
}

# Final State
final_state = {
    'universe': ['SPY', 'QQQ', 'TLT'],
    'final_allocations': {'SPY': 25.0, 'QQQ': 30.0, 'TLT': 45.0},
    'agent_reasoning': {...},
    # ... all intermediate data
}
```

## 🚀 Performance Optimization

### Caching Strategy

```python
# src/data_fetchers/macro_fetcher.py
import functools
import time

@functools.lru_cache(maxsize=128)
def cached_fred_data(api_key: str, indicator: str, periods: int):
    """Cache FRED data for 1 hour."""
    # Implementation here
    pass
```

### Memory Management

```python
# Clean up large data structures
def cleanup_state(state):
    """Remove large data structures after processing."""
    if 'etf_data' in state:
        del state['etf_data']
    if 'news' in state:
        del state['news']
    return state
```

### Parallel Processing

```python
# Use ray for parallel agent execution
import ray

@ray.remote
def parallel_agent_analysis(agent, state):
    return agent.analyze(state)

# Execute agents in parallel
futures = []
for agent in [macro_agent, geo_agent, corr_agent]:
    future = parallel_agent_analysis.remote(agent, state)
    futures.append(future)

results = ray.get(futures)
```

## 📈 Monitoring and Metrics

### Performance Metrics

```python
# src/utils/metrics.py
import time
import logging

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def time_function(self, func_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                self.metrics[func_name] = duration
                self.logger.info(f"{func_name} took {duration:.2f} seconds")
                
                return result
            return wrapper
        return decorator
```

### Error Tracking

```python
# src/utils/error_tracking.py
import logging
import traceback

class ErrorTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def track_error(self, error, context):
        self.logger.error(f"Error in {context}: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Send to monitoring service if configured
        if os.getenv('MONITORING_ENABLED'):
            self.send_to_monitoring(error, context)
```

## 🔒 Security Considerations

### API Key Management

```python
# src/utils/security.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
    
    def encrypt_api_key(self, api_key: str) -> str:
        if self.cipher:
            return self.cipher.encrypt(api_key.encode()).decode()
        return api_key
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        if self.cipher:
            return self.cipher.decrypt(encrypted_key.encode()).decode()
        return encrypted_key
```

### Input Validation

```python
# src/utils/validation.py
import re
from typing import List, Dict, Any

class InputValidator:
    @staticmethod
    def validate_etf_universe(universe: List[str]) -> List[str]:
        """Validate ETF symbols."""
        valid_etfs = []
        for etf in universe:
            if re.match(r'^[A-Z]{1,5}$', etf):
                valid_etfs.append(etf)
            else:
                logging.warning(f"Invalid ETF symbol: {etf}")
        return valid_etfs
    
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date format."""
        try:
            pd.to_datetime(date_str)
            return True
        except:
            return False
```

## 📚 Best Practices

### Code Organization

1. **Single Responsibility**: Each agent should have one clear purpose
2. **Error Handling**: Always handle exceptions gracefully
3. **Logging**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
4. **Type Hints**: Use type hints for better code documentation
5. **Documentation**: Document all public methods and classes

### Testing Strategy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test with large datasets
5. **Error Tests**: Test error handling and edge cases

### Deployment Considerations

1. **Environment Variables**: Use environment variables for configuration
2. **Secrets Management**: Never commit API keys to version control
3. **Dependency Management**: Use Poetry for reproducible builds
4. **Monitoring**: Implement comprehensive logging and monitoring
5. **Scaling**: Design for horizontal scaling if needed

---

This developer guide provides the foundation for understanding and extending the Global Macro ETF Trading System. For specific implementation details, refer to the individual source files and their docstrings.

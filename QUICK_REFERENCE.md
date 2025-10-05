# Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Basic usage
poetry run python src/main.py --universe SPY QQQ TLT GLD

# With debug logging
poetry run python src/main.py --universe SPY QQQ --debug

# Custom date
poetry run python src/main.py --universe SPY QQQ --date 2024-01-01

# Test complete workflow
poetry run python src/graph/test_complete_workflow.py
```

## 🔧 Common Development Tasks

### Adding a New Agent

1. **Create agent file**:
   ```bash
   touch src/agents/new_agent.py
   ```

2. **Implement agent class**:
   ```python
   from src.agents.base_agent import BaseAgent
   
   class NewAgent(BaseAgent):
       def analyze(self, state: dict) -> dict:
           # Implementation here
           return state
   ```

3. **Add to workflow** in `src/graph/macro_trading_graph.py`:
   ```python
   self.graph.add_node('new_agent', NewAgent().analyze)
   ```

### Adding a New Data Source

1. **Create fetcher**:
   ```bash
   touch src/data_fetchers/new_fetcher.py
   ```

2. **Integrate with MacroFetcher**:
   ```python
   def fetch_comprehensive_data(self, ...):
       new_data = self.new_fetcher.fetch_new_data(params)
       all_data["new_data"] = new_data
   ```

### Switching LLM Provider

1. **Update config** in `src/config.py`:
   ```python
   LLM_CONFIG = {
       'provider': 'openai',  # or 'anthropic', 'deepseek'
       'model': 'gpt-4o',
       'api_key_env': 'OPENAI_API_KEY'
   }
   ```

2. **Set environment variable**:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

## 🧪 Testing Commands

```bash
# Test individual agents
poetry run python src/agents/macro_economist.py
poetry run python src/agents/geopolitical_analyst.py
poetry run python src/agents/risk_manager.py
poetry run python src/agents/portfolio_agent.py

# Test allocation agents
poetry run python tests/agents/test_allocation_agents.py

# Test complete workflow
poetry run python src/graph/test_complete_workflow.py

# Test with different scenarios
poetry run python src/main.py --universe SPY QQQ --debug
poetry run python src/main.py --universe GLD SLV USO UNG DBC
```

## 🔍 Debugging Commands

```bash
# Enable debug logging
poetry run python src/main.py --universe SPY QQQ --debug

# Test data fetching
poetry run python -c "
from src.data_fetchers.macro_fetcher import MacroFetcher
fetcher = MacroFetcher()
data = fetcher.fetch_macro_data(['CPIAUCSL'])
print(data)
"

# Test LLM connectivity
poetry run python -c "
from src.agents.base_agent import BaseAgent
agent = BaseAgent('TestAgent')
response = agent.llm('Hello, world!')
print(response)
"
```

## 📊 Configuration Quick Reference

### Environment Variables

```bash
# Required
export FRED_API_KEY=your_fred_key
export DEEPSEEK_API_KEY=your_deepseek_key

# Optional
export FINLIGHT_API_KEY=your_finlight_key
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

### ETF Universe Configuration

```python
# src/config.py
ETF_UNIVERSE = [
    # Add new ETFs here
    'SPY', 'QQQ', 'TLT', 'GLD',
    'NEW_ETF'  # Your new ETF
]
```

### LLM Configuration

```python
# src/config.py
LLM_CONFIG = {
    'provider': 'deepseek',  # 'openai', 'anthropic', 'deepseek'
    'model': 'deepseek-chat',  # 'gpt-4o', 'claude-3-opus'
    'api_key_env': 'DEEPSEEK_API_KEY',
    'base_url': 'https://api.deepseek.com/v1'
}
```

## 🐛 Common Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'fredapi'"
**Solution**: 
```bash
poetry install
```

### Issue: "API key not found"
**Solution**: 
```bash
export FRED_API_KEY=your_key
export DEEPSEEK_API_KEY=your_key
```

### Issue: "No allocations generated"
**Solution**: Check LLM connectivity and API keys

### Issue: "Data fetch failed"
**Solution**: Check network connectivity and API rate limits

## 📁 File Structure Quick Reference

```
src/
├── main.py                    # Entry point
├── config.py                  # System configuration
├── agents/                    # AI agents
│   ├── base_agent.py         # Base agent class
│   ├── macro_economist.py    # Macro analysis
│   ├── geopolitical_analyst.py # Geo analysis
│   ├── risk_manager.py       # Risk assessment
│   └── portfolio_agent.py    # Portfolio optimization
├── data_fetchers/            # Data sources
│   └── macro_fetcher.py       # Unified data fetching
├── graph/                     # Workflow orchestration
│   └── macro_trading_graph.py # LangGraph workflow
└── utils/                     # Utility functions
```

## 🔄 Workflow Quick Reference

```
1. Data Fetching
   ├── FRED macro indicators
   ├── ETF price data (yfinance)
   └── Geopolitical news (Finlight.me)

2. Analysis Phase
   ├── Macro Economist → ETF scores
   └── Geopolitical Analyst → ETF scores

3. Risk Management Phase
   └── Risk Manager → Risk-adjusted scores

4. Portfolio Optimization Phase
   └── Portfolio Agent → Final allocations

5. Output
   ├── Final allocations
   ├── Comprehensive reasoning
   └── Actionable insights
```

## 🎯 Agent Quick Reference

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| Macro Economist | Economic analysis | Macro data, ETF data | ETF scores (-1 to 1) |
| Geopolitical Analyst | Geo risk analysis | News data, events | ETF scores (-1 to 1) |
| Risk Manager | Risk assessment | Macro scores, geo scores | Risk-adjusted scores |
| Portfolio Agent | Mathematical optimization | Risk-adjusted scores | Final allocations |

## 📈 Output Quick Reference

### Final Allocations Format
```python
{
    'SPY': 25.0,    # 25% allocation
    'QQQ': 30.0,    # 30% allocation
    'TLT': 20.0,    # 20% allocation
    'GLD': 25.0     # 25% allocation
}
```

### Agent Reasoning Format
```python
{
    'macro_economist': {
        'scores': {'SPY': 0.2, 'QQQ': 0.1},
        'reasoning': 'Macro analysis based on...',
        'key_factors': ['inflation', 'rates'],
        'timestamp': '2024-01-01T12:00:00'
    }
}
```

## 🚀 Performance Quick Reference

### Caching
- FRED data: 24 hours
- ETF data: 1 hour
- News data: 6 hours

### Memory Management
- Large datasets processed in chunks
- State cleaned up after workflow
- Correlation matrices computed efficiently

### Parallel Processing
- Agents can run in parallel
- Use `ray` for distributed processing
- LangGraph handles orchestration

## 🔧 Development Quick Reference

### Adding New ETF
1. Add to `ETF_UNIVERSE` in `src/config.py`
2. Test with: `poetry run python src/main.py --universe NEW_ETF`

### Adding New Agent
1. Create agent class in `src/agents/`
2. Add to workflow in `src/graph/macro_trading_graph.py`
3. Update display logic in `src/main.py`

### Adding New Data Source
1. Create fetcher in `src/data_fetchers/`
2. Integrate with `MacroFetcher`
3. Update agents to use new data

### Switching LLM Provider
1. Update `LLM_CONFIG` in `src/config.py`
2. Set environment variable
3. Test with: `poetry run python src/main.py --universe SPY QQQ --debug`

---

**Need more help?** Check the full README.md and DEVELOPER_GUIDE.md for detailed documentation.

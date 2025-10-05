# AI Global Macro Fund

A sophisticated AI-powered ETF trading system that uses multiple specialized agents to analyze macroeconomic data, geopolitical events, and risk factors to generate optimal ETF portfolio allocations. The system employs LangGraph for agent orchestration and provides comprehensive reasoning for all investment decisions.

## 🎯 System Overview

This revamped system transforms traditional portfolio management by using a streamlined set of AI agents that specialize in different aspects of macro analysis:

- **Macro Economist**: Analyzes economic indicators (GDP, inflation, unemployment, interest rates) and scores ETFs
- **Geopolitical Analyst**: Assesses geopolitical risks and opportunities from news and events
- **Risk Manager**: Combines analyst scores and adjusts for risk factors and volatility
- **Portfolio Manager**: LLM-driven portfolio synthesis with comprehensive reasoning and position limits

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   LangGraph      │    │   Output        │
│                 │    │   Workflow       │    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • FRED API      │───▶│ • Data Fetching  │───▶│ • Final         │
│ • yfinance      │    │ • Macro Analysis │    │   Allocations   │
│ • Finlight.me   │    │ • Geo Analysis   │    │ • Reasoning     │
│ • News APIs     │    │ • Risk Management│    │ • Rationale     │
└─────────────────┘    │ • Portfolio Mgmt │    │ • Insights     │
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- API Keys (see Configuration section)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-hedge-fund
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the system:**
   ```bash
   poetry run python src/main.py --universe SPY TLT GLD
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following API keys:

```env
# Required API Keys
FRED_API_KEY=your_fred_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional API Keys
FINLIGHT_API_KEY=your_finlight_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### ETF Universe Configuration

The system uses a configurable ETF universe defined in `src/config.py`:

```python
ETF_UNIVERSE = [
    # Country/Region ETFs
    'EWJ', 'EWG', 'EWU', 'EWA', 'EWC', 'EWZ', 'INDA', 'FXI', 'EZA', 'TUR', 'RSX', 'EWW',
    # Currency ETFs
    'UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA', 'FXF', 'CYB',
    # Bond ETFs
    'TLT', 'IEF', 'BND', 'TIP', 'LQD', 'HYG', 'EMB', 'PCY',
    # Stock Index ETFs
    'SPY', 'QQQ', 'VEU', 'VWO', 'VGK', 'VPL', 'ACWI',
    # Commodity ETFs
    'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'CORN', 'WEAT', 'DBA', 'PDBC', 'GSG'
]
```

### LLM Provider Configuration

Switch between different LLM providers by editing `src/config.py`:

```python
LLM_CONFIG = {
    'provider': 'deepseek',  # Options: 'openai', 'anthropic', 'deepseek', 'groq'
    'model': 'deepseek-chat',
    'api_key_env': 'DEEPSEEK_API_KEY',
    'base_url': 'https://api.deepseek.com/v1'
}
```

## 📖 Usage Examples

### Basic Usage

```bash
# Use default ETF universe
poetry run python src/main.py

# Use custom ETF universe
poetry run python src/main.py --universe SPY QQQ TLT GLD UUP

# Use specific date
poetry run python src/main.py --universe SPY QQQ --date 2024-01-01

# Enable debug logging
poetry run python src/main.py --universe SPY QQQ --debug
```

### Advanced Usage

```bash
# Full workflow with comprehensive reasoning
poetry run python src/main.py --universe SPY QQQ TLT GLD EWJ EWG FXI --debug

# Test with commodity-focused portfolio
poetry run python src/main.py --universe GLD SLV USO UNG DBC

# Test with global diversified portfolio
poetry run python src/main.py --universe SPY EWJ EWG FXI GLD TLT
```

## 🧠 Agent System Details

### 1. Macro Economist Agent (`src/agents/macro_economist.py`)

**Purpose**: Analyzes macroeconomic indicators and their impact on different asset classes.

**Key Responsibilities**:
- Analyzes FRED economic indicators (CPI, unemployment, Fed funds rate, GDP)
- Scores ETFs based on macro trends and economic cycles
- Considers inflation impact on bonds vs commodities
- Evaluates interest rate environment effects
- Provides confidence levels and detailed reasoning

**Input**: Macro economic data, ETF price data
**Output**: ETF scores from -1 (strong sell) to 1 (strong buy) with confidence and reasoning

### 2. Geopolitical Analyst Agent (`src/agents/geopolitical_analyst.py`)

**Purpose**: Assesses geopolitical risks and opportunities from news and events.

**Key Responsibilities**:
- Analyzes geopolitical news and events from Finlight.me
- Evaluates regional risks and opportunities
- Considers currency and trade impacts
- Scores country-specific and regional ETFs
- Provides confidence levels and detailed reasoning

**Input**: News data, geopolitical events
**Output**: ETF scores based on geopolitical factors with confidence and reasoning

### 3. Risk Manager Agent (`src/agents/risk_manager.py`)

**Purpose**: Combines analyst scores and adjusts for risk factors and volatility.

**Key Responsibilities**:
- Combines macro and geopolitical analyst scores
- Adjusts scores based on risk factors and volatility
- Assesses risk levels (low/medium/high) for each ETF
- Provides risk-adjusted scores with detailed reasoning
- Considers correlation and concentration risks

**Input**: Macro scores, geopolitical scores, ETF data, macro data
**Output**: Risk-adjusted scores with risk levels and reasoning

### 4. Portfolio Manager Agent (`src/agents/portfolio_manager.py`)

**Purpose**: LLM-driven portfolio synthesis with comprehensive reasoning and position limits.

**Key Responsibilities**:
- Aggregates analyst scores and risk metrics from all agents
- Uses LLM synthesis for final allocation decisions
- Enforces position limits (maximum 20% of universe)
- Provides comprehensive reasoning covering 8 analytical dimensions:
  - Analyst consensus analysis
  - Risk assessment integration
  - Score aggregation logic
  - Risk-return trade-off analysis
  - Portfolio construction rationale
  - Conviction level explanation
  - Risk management considerations
  - Market outlook integration
- Focuses on highest conviction opportunities
- Generates detailed reasoning for each allocation decision

**Input**: Aggregated analyst scores, risk metrics, detailed agent reasoning
**Output**: Final allocations with comprehensive reasoning and position limits

## 🔄 Workflow Process

The system follows a streamlined LangGraph workflow:

```
1. Data Fetching
   ├── FRED macro indicators
   ├── ETF price data (yfinance)
   └── Geopolitical news (Finlight.me)

2. Analysis Phase
   ├── Macro Economist → Scores ETFs based on macro trends
   └── Geopolitical Analyst → Scores ETFs based on geo risks

3. Risk Management Phase
   └── Risk Manager → Combines scores and adjusts for risk factors

4. Portfolio Management Phase
   └── Portfolio Manager → LLM-driven synthesis with position limits

5. Output
   ├── Final allocations with actions (buy/hold/sell)
   ├── Comprehensive reasoning from all agents
   ├── Risk assessments and confidence levels
   └── Detailed report saved to reports/ folder
```

## 📊 Data Sources

### FRED (Federal Reserve Economic Data)
- **Indicators**: CPI, Unemployment Rate, Fed Funds Rate, GDP
- **API**: `fredapi` Python package
- **Usage**: Macro economic analysis

### yfinance
- **Data**: ETF price data, returns, correlations
- **API**: `yfinance` Python package
- **Usage**: Historical analysis and correlation calculations

### Finlight.me
- **Data**: Geopolitical news and events
- **API**: REST API with API key
- **Usage**: Geopolitical risk assessment

## 🛠️ Development Guide

### Adding New Agents

1. **Create agent file** in `src/agents/`:
   ```python
   from src.agents.base_agent import BaseAgent
   
   class NewAgent(BaseAgent):
       def __init__(self, agent_name: str = "NewAgent"):
           super().__init__(agent_name)
           self.specialization = "new_analysis"
       
       def analyze(self, state: dict) -> dict:
           # Implementation here
           pass
   ```

2. **Add to workflow** in `src/graph/macro_trading_graph.py`:
   ```python
   # Add node
   self.graph.add_node('new_agent', NewAgent().analyze)
   
   # Add edges
   self.graph.add_edge('previous_node', 'new_agent')
   self.graph.add_edge('new_agent', 'next_node')
   ```

3. **Update reasoning capture**:
   ```python
   state['agent_reasoning']['new_agent'] = {
       'scores': scores,
       'reasoning': 'Detailed reasoning here',
       'key_factors': ['factor1', 'factor2'],
       'timestamp': pd.Timestamp.now().isoformat()
   }
   ```

### Adding New Data Sources

1. **Create fetcher** in `src/data_fetchers/`:
   ```python
   class NewDataFetcher:
       def fetch_new_data(self, params):
           # Implementation here
           pass
   ```

2. **Integrate with MacroFetcher**:
   ```python
   def fetch_comprehensive_data(self, ...):
       # Add new data source
       new_data = self.new_fetcher.fetch_new_data(params)
       all_data["new_data"] = new_data
   ```

3. **Update agents** to use new data source

### Modifying LLM Providers

1. **Update BaseAgent** in `src/agents/base_agent.py`:
   ```python
   def _initialize_llm(self):
       if LLM_CONFIG['provider'] == 'new_provider':
           # Add new provider initialization
           pass
   ```

2. **Update config** in `src/config.py`:
   ```python
   LLM_CONFIG = {
       'provider': 'new_provider',
       'model': 'new_model',
       'api_key_env': 'NEW_PROVIDER_API_KEY',
       'base_url': 'https://api.newprovider.com/v1'
   }
   ```

## 🧪 Testing

### Run Individual Agent Tests

```bash
# Test macro economist
poetry run python src/agents/macro_economist.py

# Test geopolitical analyst
poetry run python src/agents/geopolitical_analyst.py

# Test risk manager
poetry run python src/agents/risk_manager.py

# Test portfolio manager
poetry run python src/agents/portfolio_manager.py
```

### Run Complete Workflow Tests

```bash
# Test complete workflow
poetry run python src/graph/test_complete_workflow.py

# Test individual components
poetry run python src/agents/test_allocation_agents.py
```

### Run Integration Tests

```bash
# Test with different ETF universes
poetry run python src/main.py --universe SPY TLT GLD

# Test with single ETF
poetry run python src/main.py --universe SPY

# Test error handling
poetry run python src/main.py --universe INVALID1 INVALID2
```

## 📈 Performance Optimization

### Caching
- FRED data is cached for 24 hours
- ETF data is cached for 1 hour
- News data is cached for 6 hours

### Parallel Processing
- Agents can run in parallel where possible
- Use `ray` for distributed processing (optional)

### Memory Management
- Large datasets are processed in chunks
- Correlation matrices are computed efficiently
- State is cleaned up after each workflow run

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```bash
   # Check environment variables
   echo $FRED_API_KEY
   echo $DEEPSEEK_API_KEY
   ```

2. **Import Errors**:
   ```bash
   # Reinstall dependencies
   poetry install --no-cache
   ```

3. **Data Fetching Errors**:
   ```bash
   # Check network connectivity
   # Verify API keys are valid
   # Check rate limits
   ```

4. **LLM Response Errors**:
   ```bash
   # Check LLM API key
   # Verify model availability
   # Check rate limits
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
poetry run python src/main.py --universe SPY QQQ --debug
```

## 📚 API Reference

### Main Entry Point

```python
from src.graph.macro_trading_graph import MacroTradingGraph

# Initialize graph
graph = MacroTradingGraph(debug=True)

# Run workflow and get final allocations
result = graph.propagate(['SPY', 'TLT', 'GLD'], '2024-01-01')

# Run workflow and get detailed results with reasoning
detailed_result = graph.propagate_with_details(['SPY', 'TLT', 'GLD'], '2024-01-01')
```

### Agent Interface

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def analyze(self, state: dict) -> dict:
        # Must return updated state
        return state
```

### Data Fetcher Interface

```python
from src.data_fetchers.macro_fetcher import MacroFetcher

fetcher = MacroFetcher()
data = fetcher.fetch_comprehensive_data(etfs, indicators)
```

## 🤝 Contributing

### Adding New Agents
See the comprehensive [Agent Development Guide](AGENT_DEVELOPMENT_GUIDE.md) for detailed instructions on adding new agents to the system.

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Include unit tests

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit pull request with description

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run linting
poetry run flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph** for agent orchestration
- **FRED** for economic data
- **yfinance** for financial data
- **DeepSeek** for LLM capabilities
- **Finlight.me** for geopolitical news

## 📞 Support

For questions, issues, or contributions:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed description
4. Contact the development team

---

**Last Updated**: October 2024  
**Version**: 2.0.0 (Revamped)  
**Python**: 3.11+
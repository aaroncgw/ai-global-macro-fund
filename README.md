# Global Macro ETF Trading System

A sophisticated AI-powered trading system that uses multiple specialized agents to analyze macroeconomic data, geopolitical events, and market correlations to generate optimal ETF portfolio allocations. The system employs LangGraph for agent orchestration and provides comprehensive reasoning for all investment decisions.

## 🎯 System Overview

This system transforms traditional portfolio management by using multiple AI agents that specialize in different aspects of macro analysis:

- **Macro Economist**: Analyzes economic indicators (GDP, inflation, unemployment, interest rates)
- **Geopolitical Analyst**: Assesses geopolitical risks and opportunities from news and events
- **Correlation Specialist**: Evaluates diversification benefits and portfolio balance
- **Debate Researchers**: Bullish and bearish researchers engage in structured debates
- **Trader Agent**: Converts analysis into initial allocation proposals
- **Risk Manager**: Adjusts allocations based on risk factors and volatility
- **Portfolio Optimizer**: Uses mathematical optimization for final allocations

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   LangGraph      │    │   Output        │
│                 │    │   Workflow       │    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • FRED API      │───▶│ • Data Fetching  │───▶│ • Final         │
│ • yfinance      │    │ • Macro Analysis │    │   Allocations   │
│ • Finlight.me   │    │ • Geo Analysis   │    │ • Reasoning     │
│ • News APIs     │    │ • Correlation    │    │ • Rationale     │
└─────────────────┘    │ • Debate         │    │ • Insights     │
                       │ • Trading        │    └─────────────────┘
                       │ • Risk Mgmt      │
                       │ • Optimization   │
                       └──────────────────┘
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
   poetry run python src/main.py --universe SPY QQQ TLT GLD
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
- Fetches and analyzes FRED economic indicators (CPI, unemployment, Fed funds rate, GDP)
- Scores ETFs based on macro trends and economic cycles
- Considers inflation impact on bonds vs commodities
- Evaluates interest rate environment effects

**Input**: Macro economic data, ETF price data
**Output**: ETF scores from -1 (strong sell) to 1 (strong buy)

### 2. Geopolitical Analyst Agent (`src/agents/geopolitical_analyst.py`)

**Purpose**: Assesses geopolitical risks and opportunities from news and events.

**Key Responsibilities**:
- Analyzes geopolitical news and events
- Evaluates regional risks and opportunities
- Considers currency and trade impacts
- Scores country-specific and regional ETFs

**Input**: News data, geopolitical events
**Output**: ETF scores based on geopolitical factors

### 3. Correlation Specialist Agent (`src/agents/correlation_specialist.py`)

**Purpose**: Analyzes ETF correlations and suggests diversification scores.

**Key Responsibilities**:
- Calculates correlation matrices between ETFs
- Identifies diversification opportunities
- Suggests portfolio balance improvements
- Scores ETFs based on correlation benefits

**Input**: ETF price data, correlation matrices
**Output**: Diversification-focused ETF scores

### 4. Debate Researchers (`src/agents/debate_researchers.py`)

**Purpose**: Conducts structured debates between bullish and bearish perspectives.

**Key Components**:
- **BullishMacroResearcher**: Presents bullish macro opportunities
- **BearishMacroResearcher**: Presents bearish macro risks
- **Debate Function**: Orchestrates multi-round debates

**Input**: Analyst scores, market data
**Output**: Structured debate results and consensus

### 5. Trader Agent (`src/agents/trader_agent.py`)

**Purpose**: Converts analysis into actionable allocation proposals.

**Key Responsibilities**:
- Synthesizes analyst scores and debate results
- Proposes initial buy/sell allocations
- Balances risk and return objectives
- Creates actionable trading decisions

**Input**: Debate results, analyst scores
**Output**: Proposed ETF allocations (percentages)

### 6. Risk Manager Agent (`src/agents/risk_manager.py`)

**Purpose**: Assesses and adjusts allocations for macro risks and volatility.

**Key Responsibilities**:
- Adjusts allocations based on risk factors
- Caps volatile ETFs during high inflation
- Reduces exposure during economic uncertainty
- Applies risk management constraints

**Input**: Proposed allocations, macro data, risk factors
**Output**: Risk-adjusted allocations with reasoning

### 7. Portfolio Optimizer Agent (`src/agents/portfolio_optimizer.py`)

**Purpose**: Uses mathematical optimization for final allocation decisions.

**Key Responsibilities**:
- Performs mean-variance optimization
- Considers correlation matrices
- Applies portfolio constraints
- Generates mathematically optimal allocations

**Input**: Risk-adjusted allocations, correlation data
**Output**: Final optimized allocations

## 🔄 Workflow Process

The system follows a structured LangGraph workflow:

```
1. Data Fetching
   ├── FRED macro indicators
   ├── ETF price data (yfinance)
   └── Geopolitical news (Finlight.me)

2. Analysis Phase
   ├── Macro Economist → Scores ETFs based on macro trends
   ├── Geopolitical Analyst → Scores ETFs based on geo risks
   └── Correlation Specialist → Scores ETFs for diversification

3. Debate Phase
   ├── Bullish Researcher → Presents bullish arguments
   └── Bearish Researcher → Presents bearish counterarguments

4. Allocation Phase
   ├── Trader Agent → Proposes initial allocations
   ├── Risk Manager → Adjusts for risk factors
   └── Portfolio Optimizer → Final mathematical optimization

5. Output
   ├── Final allocations
   ├── Comprehensive reasoning
   └── Actionable insights
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

# Test correlation specialist
poetry run python src/agents/correlation_specialist.py
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
poetry run python src/graph/test_complete_workflow.py

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

# Run workflow
result = graph.propagate(['SPY', 'QQQ', 'TLT'], '2024-01-01')
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

**Last Updated**: September 2024  
**Version**: 1.0.0  
**Python**: 3.11+
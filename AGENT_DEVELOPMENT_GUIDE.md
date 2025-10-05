# Agent Development Guide

## 🎯 Overview

This guide provides comprehensive instructions for adding new agents to the AI Global Macro Fund system. The system is designed to be modular and extensible, allowing new coding agents to easily integrate specialized analysis capabilities.

## 🏗️ System Architecture

### Core Components
- **BaseAgent**: Abstract base class for all agents
- **LangGraph Workflow**: Orchestrates agent execution
- **State Management**: Shared state dictionary between agents
- **LLM Integration**: DeepSeek LLM with deterministic outputs

### Current Agent Types
1. **Signal Agents**: Analyze data and output scores (MacroEconomist, GeopoliticalAnalyst)
2. **Risk Manager**: Assess risks independently of analyst scores
3. **Portfolio Manager**: Synthesize all inputs for final allocations

## 📋 Agent Development Rules & Conventions

### 1. Agent Class Structure

```python
from src.agents.base_agent import BaseAgent
import logging

class YourNewAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_name = "YourNewAgent"
    
    def analyze(self, state: dict) -> dict:
        """
        Main analysis method - REQUIRED for all agents.
        
        Args:
            state: Shared state dictionary containing:
                - universe: List of ETF symbols
                - macro_data: Economic indicators
                - etf_data: ETF price data
                - news: Geopolitical news
                - [previous agent outputs]
        
        Returns:
            Updated state dictionary
        """
        # Your analysis logic here
        return state
```

### 2. Required Methods

#### `analyze(self, state: dict) -> dict`
- **MUST** be implemented by all agents
- **MUST** return the updated state dictionary
- **MUST** handle errors gracefully with logging
- **SHOULD** use `self.llm(prompt, response_format='json_object')` for LLM calls

### 3. State Management Rules

#### Input State Keys (Available to all agents):
- `universe`: List of ETF symbols
- `macro_data`: Economic indicators from FRED
- `etf_data`: ETF price data from Yahoo Finance
- `news`: Geopolitical news from Finlight.me
- `timestamp`: Analysis timestamp

#### Output State Keys (Agent-specific):
- **Signal Agents**: `{agent_name}_scores` (e.g., `macro_scores`, `geo_scores`)
- **Risk Manager**: `risk_metrics`
- **Portfolio Manager**: `final_allocations`

### 4. LLM Integration Rules

#### Prompt Structure:
```python
prompt = f"""
Your analysis prompt here...

EXAMPLES OF EXPECTED OUTPUT:
Example 1 - [Scenario]:
{{"ETF": {{"key": value, "key2": value2}}}}

CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
{{"ETF": {{"key": value, "key2": value2}}}}

Do not include any text before or after the JSON. Return only the JSON object.
"""

# Get LLM response with JSON format
response = self.llm(prompt, response_format='json_object')
```

#### JSON Output Requirements:
- **MUST** be valid JSON
- **MUST** include all ETFs in universe
- **MUST** follow exact format specified in examples
- **MUST** use `response_format='json_object'`

### 5. Error Handling Rules

```python
def analyze(self, state: dict) -> dict:
    try:
        # Your analysis logic
        result = self._perform_analysis(state)
        state[f'{self.agent_name.lower()}_scores'] = result
        return state
    except Exception as e:
        logger.error(f"{self.agent_name} analysis failed: {e}")
        # Provide fallback values
        fallback_result = self._create_fallback_result(state['universe'])
        state[f'{self.agent_name.lower()}_scores'] = fallback_result
        return state
```

### 6. Logging Requirements

```python
import logging
logger = logging.getLogger(__name__)

# Log successful completion
logger.info(f"{self.agent_name} analysis completed for {len(state['universe'])} ETFs")

# Log errors
logger.error(f"{self.agent_name} analysis failed: {error_message}")
```

## 🚀 Adding a New Agent

### Step 1: Create Agent File

```bash
# Create new agent file
touch src/agents/your_new_agent.py
```

### Step 2: Implement Agent Class

```python
"""
Your New Agent for Global Macro ETF Trading System

This agent provides [specific analysis capability].
"""

import logging
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class YourNewAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_name = "YourNewAgent"
    
    def analyze(self, state: dict) -> dict:
        """
        Perform [specific analysis] on ETF universe.
        
        Args:
            state: Shared state dictionary
            
        Returns:
            Updated state with your_new_scores
        """
        try:
            universe = state['universe']
            # Add your data sources here
            your_data = state.get('your_data', {})
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(universe, your_data)
            
            # Get LLM response
            response = self.llm(prompt, response_format='json_object')
            
            # Parse response
            scores = self._parse_scores(response, universe)
            
            # Update state
            state['your_new_scores'] = scores
            
            logger.info(f"{self.agent_name} analysis completed for {len(universe)} ETFs")
            return state
            
        except Exception as e:
            logger.error(f"{self.agent_name} analysis failed: {e}")
            # Provide fallback
            fallback_scores = self._create_fallback_scores(universe)
            state['your_new_scores'] = fallback_scores
            return state
    
    def _create_analysis_prompt(self, universe: list, your_data: dict) -> str:
        """Create analysis prompt with examples."""
        return f"""
        As a [your specialization], analyze the following data and score each ETF from -1 (strong sell) to 1 (strong buy).
        
        ETFs TO ANALYZE: {', '.join(universe)}
        
        YOUR DATA: {your_data}
        
        EXAMPLES OF EXPECTED OUTPUT:
        Example 1 - [Scenario]:
        {{"SPY": {{"score": 0.5, "confidence": 0.8, "reason": "Detailed explanation"}}, "TLT": {{"score": -0.3, "confidence": 0.6, "reason": "Detailed explanation"}}}}
        
        CRITICAL: Output only valid JSON dict with no extra text, explanations, or formatting:
        {{"ETF": {{"score": -1 to 1, "confidence": 0-1, "reason": "detailed explanation"}}}}
        
        Do not include any text before or after the JSON. Return only the JSON object.
        """
    
    def _parse_scores(self, response: str, universe: list) -> dict:
        """Parse LLM response into structured scores."""
        try:
            import json
            scores = json.loads(response)
            
            # Validate and structure scores
            validated_scores = {}
            for etf in universe:
                if etf in scores and isinstance(scores[etf], dict):
                    etf_data = scores[etf]
                    score = float(etf_data.get('score', 0.0))
                    confidence = float(etf_data.get('confidence', 0.5))
                    reason = str(etf_data.get('reason', 'No reasoning provided'))
                    
                    # Clamp values to valid ranges
                    score = max(-1.0, min(1.0, score))
                    confidence = max(0.0, min(1.0, confidence))
                    
                    validated_scores[etf] = {
                        'score': score,
                        'confidence': confidence,
                        'reason': reason
                    }
                else:
                    # Default values for missing ETFs
                    validated_scores[etf] = {
                        'score': 0.0,
                        'confidence': 0.0,
                        'reason': 'No analysis available'
                    }
            
            return validated_scores
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse scores from response: {e}")
            return self._create_fallback_scores(universe)
    
    def _create_fallback_scores(self, universe: list) -> dict:
        """Create fallback scores when analysis fails."""
        return {etf: {
            'score': 0.0,
            'confidence': 0.0,
            'reason': f'Analysis failed - {self.agent_name} error'
        } for etf in universe}

# Test section
if __name__ == "__main__":
    # Test the agent
    agent = YourNewAgent()
    
    # Create test state
    test_state = {
        'universe': ['SPY', 'TLT', 'GLD'],
        'your_data': {'test': 'data'},
        'macro_data': {},
        'etf_data': None,
        'news': []
    }
    
    # Run analysis
    result = agent.analyze(test_state)
    print(f"Analysis completed: {result.get('your_new_scores', {})}")
```

### Step 3: Add to Configuration

Update `src/config.py`:

```python
# Add to SIGNAL_AGENTS list
SIGNAL_AGENTS = [
    MacroEconomistAgent,
    GeopoliticalAnalystAgent,
    YourNewAgent,  # Add your agent here
]

# Add to AGENT_CONFIG
AGENT_CONFIG = {
    'macro_economist': {'enabled': True, 'weight': 0.4},
    'geopolitical_analyst': {'enabled': True, 'weight': 0.3},
    'your_new_agent': {'enabled': True, 'weight': 0.3},  # Add your agent
    'risk_manager': {'enabled': True, 'weight': 1.0},
    'portfolio_manager': {'enabled': True, 'weight': 1.0}
}
```

### Step 4: Update Workflow (if needed)

The system automatically includes new signal agents in the workflow. For special cases, update `src/graph/macro_trading_graph.py`:

```python
# The system automatically handles new signal agents
# No manual workflow updates needed for standard signal agents
```

### Step 5: Create Tests

Create `tests/agents/test_your_new_agent.py`:

```python
"""
Test Your New Agent
"""

import pytest
from src.agents.your_new_agent import YourNewAgent

def test_your_new_agent():
    """Test your new agent functionality."""
    agent = YourNewAgent()
    
    # Create test state
    test_state = {
        'universe': ['SPY', 'TLT', 'GLD'],
        'your_data': {'test': 'data'},
        'macro_data': {},
        'etf_data': None,
        'news': []
    }
    
    # Run analysis
    result = agent.analyze(test_state)
    
    # Verify results
    assert 'your_new_scores' in result
    assert len(result['your_new_scores']) == 3
    
    for etf in ['SPY', 'TLT', 'GLD']:
        assert etf in result['your_new_scores']
        score_data = result['your_new_scores'][etf]
        assert 'score' in score_data
        assert 'confidence' in score_data
        assert 'reason' in score_data
        assert -1 <= score_data['score'] <= 1
        assert 0 <= score_data['confidence'] <= 1

if __name__ == "__main__":
    test_your_new_agent()
    print("✅ Your new agent test passed!")
```

## 🧪 Testing Your Agent

### Individual Agent Test
```bash
poetry run python src/agents/your_new_agent.py
```

### Integration Test
```bash
poetry run python src/main.py --universe SPY TLT GLD
```

### Full System Test
```bash
poetry run python tests/agents/test_your_new_agent.py
```

## 📊 Agent Output Standards

### Signal Agents Output Format
```json
{
  "ETF": {
    "score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "reason": "Detailed explanation of reasoning"
  }
}
```

### Risk Manager Output Format
```json
{
  "ETF": {
    "risk_level": "low/medium/high",
    "volatility": 0.0 to 1.0,
    "reason": "Detailed risk assessment"
  }
}
```

### Portfolio Manager Output Format
```json
{
  "ETF": {
    "action": "buy/sell/hold",
    "allocation": 0.0 to 1.0,
    "reason": "Comprehensive allocation reasoning"
  }
}
```

## 🔧 Common Patterns

### 1. Data Access Pattern
```python
def analyze(self, state: dict) -> dict:
    # Access required data
    universe = state['universe']
    macro_data = state.get('macro_data', {})
    etf_data = state.get('etf_data')
    news = state.get('news', [])
    
    # Your analysis logic
    pass
```

### 2. Error Handling Pattern
```python
def analyze(self, state: dict) -> dict:
    try:
        # Analysis logic
        result = self._perform_analysis(state)
        state[f'{self.agent_name.lower()}_scores'] = result
        return state
    except Exception as e:
        logger.error(f"{self.agent_name} analysis failed: {e}")
        fallback = self._create_fallback_result(state['universe'])
        state[f'{self.agent_name.lower()}_scores'] = fallback
        return state
```

### 3. JSON Parsing Pattern
```python
def _parse_response(self, response: str, universe: list) -> dict:
    try:
        import json
        scores = json.loads(response)
        
        validated_scores = {}
        for etf in universe:
            if etf in scores and isinstance(scores[etf], dict):
                etf_data = scores[etf]
                validated_scores[etf] = {
                    'score': max(-1.0, min(1.0, float(etf_data.get('score', 0.0)))),
                    'confidence': max(0.0, min(1.0, float(etf_data.get('confidence', 0.5)))),
                    'reason': str(etf_data.get('reason', 'No reasoning provided'))
                }
            else:
                validated_scores[etf] = {
                    'score': 0.0,
                    'confidence': 0.0,
                    'reason': 'No analysis available'
                }
        
        return validated_scores
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse response: {e}")
        return self._create_fallback_scores(universe)
```

## 🚨 Important Rules

### 1. State Management
- **NEVER** modify input state directly
- **ALWAYS** return updated state dictionary
- **ALWAYS** include all ETFs in universe in output

### 2. Error Handling
- **ALWAYS** handle exceptions gracefully
- **ALWAYS** provide fallback values
- **ALWAYS** log errors with context

### 3. LLM Integration
- **ALWAYS** use `response_format='json_object'`
- **ALWAYS** include examples in prompts
- **ALWAYS** validate JSON responses

### 4. Testing
- **ALWAYS** test individual agent
- **ALWAYS** test integration with system
- **ALWAYS** verify output format compliance

## 📚 Additional Resources

- **BaseAgent**: `src/agents/base_agent.py`
- **Configuration**: `src/config.py`
- **Workflow**: `src/graph/macro_trading_graph.py`
- **Examples**: `src/agents/macro_economist.py`, `src/agents/geopolitical_analyst.py`

## 🤝 Getting Help

1. Check existing agent implementations for patterns
2. Review test files for usage examples
3. Use debug logging to troubleshoot issues
4. Follow the established conventions strictly

---

**Remember**: The system is designed to be modular and extensible. Follow these conventions, and your new agent will integrate seamlessly with the existing workflow!

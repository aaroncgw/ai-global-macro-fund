"""
Base Agent for LLM Flexibility in Global Macro ETF Trading System

This module provides a base agent class that abstracts LLM interactions
and allows for easy switching between different LLM providers via configuration.
All other agents in the system extend this base class for LLM calls.
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Load environment variables
load_dotenv()

# Import configuration
from src.config import LLM_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base agent class that provides LLM abstraction and flexibility.
    
    This class handles LLM provider initialization and switching,
    allowing the system to use different LLM providers (DeepSeek, OpenAI, Anthropic, etc.)
    by simply changing the LLM_CONFIG in the configuration file.
    """
    
    def __init__(self, agent_name: str = "BaseAgent"):
        """
        Initialize the base agent with LLM provider based on configuration.
        
        Args:
            agent_name: Name of the agent for logging purposes
        """
        self.agent_name = agent_name
        self.llm_model = None
        self.provider = LLM_CONFIG.get('provider', 'deepseek')
        
        # Initialize LLM based on provider configuration
        self._initialize_llm()
        
        logger.info(f"{self.agent_name} initialized with {self.provider} provider")
    
    def _initialize_llm(self):
        """Initialize the LLM model based on the configured provider."""
        try:
            if self.provider == 'deepseek':
                self._initialize_deepseek()
            elif self.provider == 'openai':
                self._initialize_openai()
            elif self.provider == 'anthropic':
                self._initialize_anthropic()
            elif self.provider == 'google':
                self._initialize_google()
            else:
                logger.warning(f"Unknown provider '{self.provider}', falling back to DeepSeek")
                self._initialize_deepseek()
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider '{self.provider}': {e}")
            logger.info("Falling back to DeepSeek provider")
            self._initialize_deepseek()
    
    def _initialize_deepseek(self):
        """Initialize DeepSeek LLM provider."""
        api_key = os.getenv(LLM_CONFIG.get('api_key_env', 'DEEPSEEK_API_KEY'))
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        self.llm_model = ChatOpenAI(
            api_key=api_key,
            base_url=LLM_CONFIG.get('base_url', 'https://api.deepseek.com/v1'),
            model=LLM_CONFIG.get('model', 'deepseek-chat'),
            temperature=LLM_CONFIG.get('temperature', 0),
            seed=LLM_CONFIG.get('seed', 42),
            max_tokens=LLM_CONFIG.get('max_tokens', 1500),
            top_p=LLM_CONFIG.get('top_p', 1.0),
            frequency_penalty=LLM_CONFIG.get('frequency_penalty', 0.0),
            presence_penalty=LLM_CONFIG.get('presence_penalty', 0.0)
        )
        logger.info("DeepSeek LLM initialized successfully")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM provider."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm_model = ChatOpenAI(
            api_key=api_key,
            model=LLM_CONFIG.get('model', 'gpt-4'),
            temperature=LLM_CONFIG.get('temperature', 0),
            seed=LLM_CONFIG.get('seed', 42),
            max_tokens=LLM_CONFIG.get('max_tokens', 1500),
            top_p=LLM_CONFIG.get('top_p', 1.0),
            frequency_penalty=LLM_CONFIG.get('frequency_penalty', 0.0),
            presence_penalty=LLM_CONFIG.get('presence_penalty', 0.0)
        )
        logger.info("OpenAI LLM initialized successfully")
    
    def _initialize_anthropic(self):
        """Initialize Anthropic LLM provider."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.llm_model = ChatAnthropic(
            api_key=api_key,
            model=LLM_CONFIG.get('model', 'claude-3-sonnet-20240229'),
            temperature=LLM_CONFIG.get('temperature', 0),
            max_tokens=LLM_CONFIG.get('max_tokens', 1500)
        )
        logger.info("Anthropic LLM initialized successfully")
    
    def _initialize_google(self):
        """Initialize Google LLM provider."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.llm_model = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=LLM_CONFIG.get('model', 'gemini-pro'),
            temperature=LLM_CONFIG.get('temperature', 0),
            max_output_tokens=LLM_CONFIG.get('max_tokens', 1500)
        )
        logger.info("Google LLM initialized successfully")
    
    def llm(self, prompt: str, response_format: str = None) -> str:
        """
        Call the LLM with a text prompt and return the response.
        
        Args:
            prompt: Text prompt to send to the LLM
            response_format: Optional response format ('json_object' for structured JSON)
            
        Returns:
            LLM response as string
        """
        try:
            if response_format == 'json_object':
                # Use structured output for JSON responses
                response = self.llm_model.invoke(prompt, response_format={'type': 'json_object'})
            else:
                response = self.llm_model.invoke(prompt)
            
            # Log prompt and response for debugging
            logger.info(f'LLM Call - Agent: {self.agent_name}')
            logger.info(f'Prompt: {prompt[:100]}...')
            logger.info(f'Output: {response.content[:100]}...')
            
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}"
    
    def llm_with_structured_output(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the LLM with structured output format.
        
        Args:
            prompt: Text prompt to send to the LLM
            response_format: Dictionary defining the expected response structure
            
        Returns:
            Structured response as dictionary
        """
        try:
            # For now, return the text response wrapped in a dictionary
            # In a full implementation, this would use Pydantic models or JSON schema
            response = self.llm(prompt)
            return {
                'content': response,
                'format': response_format,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Structured LLM call failed: {e}")
            return {
                'content': f"Error: {str(e)}",
                'format': response_format,
                'status': 'error'
            }
    
    def get_provider_info(self) -> Dict[str, str]:
        """
        Get information about the current LLM provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            'provider': self.provider,
            'model': LLM_CONFIG.get('model', 'unknown'),
            'base_url': LLM_CONFIG.get('base_url', 'default'),
            'agent_name': self.agent_name
        }
    
    def switch_provider(self, new_provider: str):
        """
        Switch to a different LLM provider at runtime.
        
        Args:
            new_provider: Name of the new provider to switch to
        """
        old_provider = self.provider
        self.provider = new_provider
        
        try:
            self._initialize_llm()
            logger.info(f"Successfully switched from {old_provider} to {new_provider}")
        except Exception as e:
            logger.error(f"Failed to switch to {new_provider}: {e}")
            self.provider = old_provider
            raise
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by subclasses.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Analysis results
        """
        pass


class MacroAgent(BaseAgent):
    """
    Concrete implementation of BaseAgent for macro analysis.
    
    This class provides a template for macro-focused agents that can be
    extended by specific agent implementations.
    """
    
    def __init__(self, agent_name: str = "MacroAgent"):
        """Initialize the macro agent."""
        super().__init__(agent_name)
        self.analysis_type = "macro"
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform macro analysis on the provided data.
        
        Args:
            data: Dictionary containing macro data (ETFs, indicators, news)
            
        Returns:
            Analysis results with macro insights
        """
        try:
            # Extract relevant data
            etfs = data.get('etfs', [])
            macro_indicators = data.get('macro_indicators', {})
            news_data = data.get('news_data', [])
            
            # Create analysis prompt
            prompt = self._create_macro_analysis_prompt(etfs, macro_indicators, news_data)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse and structure the response
            analysis_result = {
                'agent_name': self.agent_name,
                'analysis_type': self.analysis_type,
                'etfs_analyzed': etfs,
                'indicators_used': list(macro_indicators.keys()),
                'news_articles_count': len(news_data),
                'llm_response': response,
                'timestamp': data.get('timestamp', 'unknown')
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return {
                'agent_name': self.agent_name,
                'analysis_type': self.analysis_type,
                'error': str(e),
                'status': 'failed'
            }
    
    def _create_macro_analysis_prompt(self, etfs: list, indicators: dict, news: list) -> str:
        """
        Create a prompt for macro analysis.
        
        Args:
            etfs: List of ETF tickers
            indicators: Dictionary of macro indicators
            news: List of news articles
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        As a global macro analyst, analyze the following data and provide insights:
        
        ETFs to analyze: {', '.join(etfs)}
        
        Macro Indicators:
        {self._format_indicators(indicators)}
        
        News Articles ({len(news)} articles):
        {self._format_news(news)}
        
        Please provide:
        1. Overall macro environment assessment
        2. Key risks and opportunities
        3. Recommended ETF allocations
        4. Risk factors to monitor
        
        Focus on global macro trends, economic cycles, and geopolitical factors.
        """
        return prompt
    
    def _format_indicators(self, indicators: dict) -> str:
        """Format macro indicators for the prompt."""
        if not indicators:
            return "No macro indicators available"
        
        formatted = []
        for indicator, data in indicators.items():
            if 'error' not in data:
                latest_value = data.get('latest_value', 'N/A')
                periods = data.get('periods', 0)
                formatted.append(f"- {indicator}: {latest_value} ({periods} periods)")
            else:
                formatted.append(f"- {indicator}: Error - {data.get('error', 'Unknown')}")
        
        return '\n'.join(formatted)
    
    def _format_news(self, news: list) -> str:
        """Format news articles for the prompt."""
        if not news:
            return "No news articles available"
        
        formatted = []
        for i, article in enumerate(news[:5]):  # Limit to first 5 articles
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary')
            formatted.append(f"{i+1}. {title}\n   {summary[:200]}...")
        
        return '\n'.join(formatted)


# Example usage and testing
if __name__ == "__main__":
    print("Base Agent Test Suite")
    print("="*50)
    
    # Test base agent initialization
    try:
        agent = MacroAgent("TestAgent")
        print(f"✓ Base agent initialized successfully")
        print(f"  Provider: {agent.get_provider_info()['provider']}")
        print(f"  Model: {agent.get_provider_info()['model']}")
        
        # Test LLM call
        test_prompt = "What are the key factors affecting global macro markets today?"
        response = agent.llm(test_prompt)
        print(f"✓ LLM call successful")
        print(f"  Response length: {len(response)} characters")
        
        # Test analysis
        test_data = {
            'etfs': ['SPY', 'QQQ', 'TLT'],
            'macro_indicators': {'CPIAUCSL': {'latest_value': 300.0, 'periods': 12}},
            'news_data': [{'title': 'Test News', 'summary': 'Test summary'}],
            'timestamp': '2024-01-01T00:00:00'
        }
        
        analysis_result = agent.analyze(test_data)
        print(f"✓ Analysis completed")
        print(f"  ETFs analyzed: {analysis_result.get('etfs_analyzed', [])}")
        print(f"  Analysis type: {analysis_result.get('analysis_type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Base agent test completed!")

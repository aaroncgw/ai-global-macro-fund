"""
LLM Provider Configuration Examples

This module demonstrates how to switch between different LLM providers
by modifying the LLM_CONFIG in the configuration file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def show_current_config():
    """Display the current LLM configuration."""
    from src.config import LLM_CONFIG
    
    print("Current LLM Configuration:")
    print("="*40)
    for key, value in LLM_CONFIG.items():
        print(f"  {key}: {value}")
    print()


def show_provider_examples():
    """Show examples of different LLM provider configurations."""
    
    print("LLM Provider Configuration Examples")
    print("="*50)
    
    # DeepSeek Configuration (Default)
    print("1. DeepSeek Configuration (Current Default):")
    print("""
    LLM_CONFIG = {
        'provider': 'deepseek',
        'model': 'deepseek-chat',
        'api_key_env': 'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com/v1',
        'temperature': 0.7,
        'max_tokens': 4000,
    }
    """)
    
    # OpenAI Configuration
    print("2. OpenAI Configuration:")
    print("""
    LLM_CONFIG = {
        'provider': 'openai',
        'model': 'gpt-4',
        'api_key_env': 'OPENAI_API_KEY',
        'base_url': 'https://api.openai.com/v1',
        'temperature': 0.7,
        'max_tokens': 4000,
    }
    """)
    
    # Anthropic Configuration
    print("3. Anthropic Configuration:")
    print("""
    LLM_CONFIG = {
        'provider': 'anthropic',
        'model': 'claude-3-sonnet-20240229',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'base_url': 'https://api.anthropic.com',
        'temperature': 0.7,
        'max_tokens': 4000,
    }
    """)
    
    # Google Configuration
    print("4. Google Configuration:")
    print("""
    LLM_CONFIG = {
        'provider': 'google',
        'model': 'gemini-pro',
        'api_key_env': 'GOOGLE_API_KEY',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'temperature': 0.7,
        'max_tokens': 4000,
    }
    """)


def test_provider_switching():
    """Test switching between different LLM providers."""
    print("Testing Provider Switching")
    print("="*30)
    
    try:
        from src.agents.base_agent import MacroAgent
        
        # Test with current configuration
        agent = MacroAgent("TestAgent")
        current_provider = agent.get_provider_info()['provider']
        print(f"✓ Current provider: {current_provider}")
        
        # Test LLM call
        test_prompt = "What is the current state of global macro markets?"
        response = agent.llm(test_prompt)
        print(f"✓ LLM call successful (response length: {len(response)}")
        
        # Show how to switch providers
        print("\nTo switch providers:")
        print("1. Edit src/config.py")
        print("2. Change the LLM_CONFIG dictionary")
        print("3. Restart the application")
        print("4. The BaseAgent will automatically use the new provider")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def show_environment_setup():
    """Show how to set up environment variables for different providers."""
    
    print("Environment Variable Setup")
    print("="*30)
    
    print("Required environment variables for each provider:")
    print()
    
    print("DeepSeek:")
    print("  DEEPSEEK_API_KEY=your_deepseek_api_key")
    print()
    
    print("OpenAI:")
    print("  OPENAI_API_KEY=your_openai_api_key")
    print()
    
    print("Anthropic:")
    print("  ANTHROPIC_API_KEY=your_anthropic_api_key")
    print()
    
    print("Google:")
    print("  GOOGLE_API_KEY=your_google_api_key")
    print()
    
    print("Example .env file:")
    print("""
    # LLM Provider API Keys
    DEEPSEEK_API_KEY=your_deepseek_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    GOOGLE_API_KEY=your_google_api_key_here
    
    # Other API Keys
    FRED_API_KEY=your_fred_api_key_here
    FINLIGHT_API_KEY=your_finlight_api_key_here
    """)


def show_usage_examples():
    """Show examples of how to use the base agent in different scenarios."""
    
    print("Usage Examples")
    print("="*20)
    
    print("1. Basic Agent Usage:")
    print("""
    from src.agents.base_agent import MacroAgent
    
    # Initialize agent
    agent = MacroAgent("MyAgent")
    
    # Make LLM call
    response = agent.llm("Analyze the current macro environment")
    print(response)
    """)
    
    print("2. Custom Agent Implementation:")
    print("""
    from src.agents.base_agent import BaseAgent
    
    class MyCustomAgent(BaseAgent):
        def __init__(self):
            super().__init__("MyCustomAgent")
        
        def analyze(self, data):
            prompt = self._create_prompt(data)
            response = self.llm(prompt)
            return self._parse_response(response)
    """)
    
    print("3. Provider Switching at Runtime:")
    print("""
    from src.agents.base_agent import MacroAgent
    
    agent = MacroAgent("MyAgent")
    
    # Switch to OpenAI
    agent.switch_provider("openai")
    
    # Continue using the same interface
    response = agent.llm("Same prompt, different provider")
    """)


if __name__ == "__main__":
    print("LLM Provider Configuration Guide")
    print("="*50)
    
    # Show current configuration
    show_current_config()
    
    # Show provider examples
    show_provider_examples()
    
    # Test provider switching
    test_provider_switching()
    
    # Show environment setup
    show_environment_setup()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "="*50)
    print("Configuration guide completed!")

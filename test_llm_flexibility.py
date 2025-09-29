#!/usr/bin/env python3
"""
Test LLM Provider Flexibility

This script tests the ability to switch between different LLM providers
by modifying the configuration and verifying the system works correctly.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import LLM_CONFIG
from src.agents.macro_economist import MacroEconomistAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_flexibility():
    """Test LLM provider flexibility."""
    print("🤖 Testing LLM Provider Flexibility")
    print("=" * 50)
    
    # Test configurations for different providers
    test_configs = [
        {
            'name': 'DeepSeek',
            'config': {
                'provider': 'deepseek',
                'model': 'deepseek-chat',
                'api_key_env': 'DEEPSEEK_API_KEY',
                'base_url': 'https://api.deepseek.com/v1'
            }
        },
        {
            'name': 'OpenAI',
            'config': {
                'provider': 'openai',
                'model': 'gpt-4o',
                'api_key_env': 'OPENAI_API_KEY',
                'base_url': 'https://api.openai.com/v1'
            }
        },
        {
            'name': 'Anthropic',
            'config': {
                'provider': 'anthropic',
                'model': 'claude-3-opus-20240229',
                'api_key_env': 'ANTHROPIC_API_KEY',
                'base_url': 'https://api.anthropic.com'
            }
        },
        {
            'name': 'Google',
            'config': {
                'provider': 'google',
                'model': 'gemini-pro',
                'api_key_env': 'GOOGLE_API_KEY',
                'base_url': 'https://generativelanguage.googleapis.com/v1'
            }
        }
    ]
    
    results = {
        'total_providers': len(test_configs),
        'supported_providers': [],
        'unsupported_providers': [],
        'flexibility_score': 0
    }
    
    for test_config in test_configs:
        provider_name = test_config['name']
        config = test_config['config']
        
        print(f"\n🧪 Testing {provider_name}...")
        
        try:
            # Create a test agent with the configuration
            agent = MacroEconomistAgent("TestAgent")
            
            # Test if the provider can be initialized
            # (This doesn't actually call the API, just tests configuration)
            agent.switch_provider(config)
            
            # Get provider info
            provider_info = agent.get_provider_info()
            
            print(f"   ✅ {provider_name} configuration valid")
            print(f"   📋 Provider info: {provider_info}")
            
            results['supported_providers'].append(provider_name)
            
        except Exception as e:
            print(f"   ❌ {provider_name} configuration failed: {e}")
            results['unsupported_providers'].append({
                'provider': provider_name,
                'error': str(e)
            })
    
    # Calculate flexibility score
    results['flexibility_score'] = (
        len(results['supported_providers']) / len(test_configs) * 100
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 LLM FLEXIBILITY RESULTS")
    print("=" * 50)
    print(f"Supported providers: {len(results['supported_providers'])}/{len(test_configs)}")
    print(f"Flexibility score: {results['flexibility_score']:.1f}%")
    
    if results['supported_providers']:
        print(f"\n✅ Supported providers:")
        for provider in results['supported_providers']:
            print(f"   - {provider}")
    
    if results['unsupported_providers']:
        print(f"\n❌ Unsupported providers:")
        for provider_info in results['unsupported_providers']:
            print(f"   - {provider_info['provider']}: {provider_info['error']}")
    
    # Test configuration switching
    print(f"\n🔄 Testing configuration switching...")
    try:
        # Test switching from current config to a different one
        current_config = LLM_CONFIG.copy()
        print(f"   Current config: {current_config['provider']}")
        
        # Test switching to OpenAI (if not already)
        if current_config['provider'] != 'openai':
            test_agent = MacroEconomistAgent("SwitchTestAgent")
            openai_config = {
                'provider': 'openai',
                'model': 'gpt-4o',
                'api_key_env': 'OPENAI_API_KEY',
                'base_url': 'https://api.openai.com/v1'
            }
            
            test_agent.switch_provider(openai_config)
            new_info = test_agent.get_provider_info()
            print(f"   ✅ Successfully switched to: {new_info['provider']}")
        else:
            print(f"   ℹ️  Already using OpenAI configuration")
        
        results['configuration_switching'] = True
        
    except Exception as e:
        print(f"   ❌ Configuration switching failed: {e}")
        results['configuration_switching'] = False
    
    # Save results
    results_file = Path("llm_flexibility_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Results saved to: {results_file}")
    
    return results

def test_configuration_changes():
    """Test how to change LLM configuration."""
    print("\n🔧 LLM Configuration Guide")
    print("=" * 50)
    
    config_file = Path("src/config.py")
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
        
        print("To switch LLM providers, edit src/config.py:")
        print("\n1. For OpenAI:")
        print("   LLM_CONFIG = {")
        print("       'provider': 'openai',")
        print("       'model': 'gpt-4o',")
        print("       'api_key_env': 'OPENAI_API_KEY',")
        print("       'base_url': 'https://api.openai.com/v1'")
        print("   }")
        
        print("\n2. For Anthropic:")
        print("   LLM_CONFIG = {")
        print("       'provider': 'anthropic',")
        print("       'model': 'claude-3-opus-20240229',")
        print("       'api_key_env': 'ANTHROPIC_API_KEY',")
        print("       'base_url': 'https://api.anthropic.com'")
        print("   }")
        
        print("\n3. For DeepSeek (default):")
        print("   LLM_CONFIG = {")
        print("       'provider': 'deepseek',")
        print("       'model': 'deepseek-chat',")
        print("       'api_key_env': 'DEEPSEEK_API_KEY',")
        print("       'base_url': 'https://api.deepseek.com/v1'")
        print("   }")
        
        print("\n4. Set the corresponding environment variable:")
        print("   export OPENAI_API_KEY=your_key_here")
        print("   export ANTHROPIC_API_KEY=your_key_here")
        print("   export DEEPSEEK_API_KEY=your_key_here")

def main():
    """Main function to test LLM flexibility."""
    print("🤖 Global Macro ETF Trading System - LLM Flexibility Test")
    print("=" * 70)
    
    try:
        # Test LLM flexibility
        results = test_llm_flexibility()
        
        # Show configuration guide
        test_configuration_changes()
        
        print("\n✅ LLM flexibility testing completed!")
        
        return 0 if results['flexibility_score'] > 50 else 1
        
    except Exception as e:
        print(f"❌ LLM flexibility test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
Example of how to extend the ETF universe for different trading strategies.

This file demonstrates how users can modify the configuration to add new ETFs
or change the system behavior without modifying the core codebase.
"""

# Example 1: Adding crypto exposure
# To add crypto ETFs, simply append to the ETF_UNIVERSE list:
CRYPTO_ETFS = ['BITO', 'ETHE', 'GBTC']  # Bitcoin, Ethereum, Grayscale Bitcoin

# Example 2: Adding sector-specific ETFs
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI']  # Tech, Financials, Energy, Healthcare, Industrials

# Example 3: Adding thematic ETFs
THEMATIC_ETFS = ['ARKK', 'ICLN', 'PBW']  # Innovation, Clean Energy, Clean Energy

# Example 4: Custom ETF universe for specific strategy
CUSTOM_ETF_UNIVERSE = [
    # Core holdings
    'SPY', 'QQQ', 'IWM',  # US equity exposure
    
    # International diversification
    'EFA', 'EEM', 'VWO',  # Developed and emerging markets
    
    # Fixed income
    'TLT', 'IEF', 'BND',  # Treasury and corporate bonds
    
    # Commodities
    'GLD', 'SLV', 'USO',  # Gold, silver, oil
    
    # Currencies
    'UUP', 'FXE', 'FXY',  # Dollar, euro, yen
]

# Example 5: Switching LLM provider
# To use OpenAI instead of DeepSeek:
OPENAI_LLM_CONFIG = {
    'provider': 'openai',
    'model': 'gpt-4',
    'api_key_env': 'OPENAI_API_KEY',
    'base_url': 'https://api.openai.com/v1',
    'temperature': 0.7,
    'max_tokens': 4000,
}

# Example 6: Custom risk parameters for aggressive strategy
AGGRESSIVE_RISK_CONFIG = {
    'max_position_size': 0.25,  # Allow up to 25% in single ETF
    'max_category_exposure': 0.60,  # Allow up to 60% in any category
    'max_leverage': 2.0,  # Allow 2x leverage
    'stop_loss_threshold': 0.15,  # 15% stop loss
    'rebalance_threshold': 0.08,  # Rebalance when 8% drift
}

# Example 7: Custom agent configuration for specific focus
FOCUSED_AGENT_CONFIG = {
    'enabled_agents': [
        'stanley_druckenmiller',  # Only macro analysis
        'risk_manager',           # Risk management
    ],
    'agent_weights': {
        'stanley_druckenmiller': 0.70,  # Heavy emphasis on macro
        'risk_manager': 0.30,            # Risk management
    },
    'consensus_threshold': 0.50,  # Lower threshold for faster decisions
}

# Example 8: How to add new ETFs to the system
# Step 1: Add to ETF_UNIVERSE list in config.py
# Step 2: Optionally add to ETF_CATEGORIES for grouping
# Step 3: The system will automatically handle the new ETFs

# Example usage:
if __name__ == "__main__":
    print("This file demonstrates how to extend the ETF trading system.")
    print("To use these examples:")
    print("1. Copy the desired configuration to src/config.py")
    print("2. Modify the values as needed")
    print("3. The system will automatically use the new configuration")
    print("\nExample ETF universes:")
    print(f"Crypto ETFs: {CRYPTO_ETFS}")
    print(f"Sector ETFs: {SECTOR_ETFS}")
    print(f"Thematic ETFs: {THEMATIC_ETFS}")
    print(f"Custom universe: {CUSTOM_ETF_UNIVERSE}")

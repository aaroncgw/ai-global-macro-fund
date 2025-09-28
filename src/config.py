"""
Global Macro ETF Trading System Configuration

This file defines the ETF universe and system configuration for a global macro trading system.
The system is designed to be generic and configurable - users can add new ETFs by editing
the ETF_UNIVERSE list, and the system will handle them dynamically without code changes.
"""

# Global Macro ETF Universe
# This list captures opportunities across economic cycles, policy shifts, and geopolitics
ETF_UNIVERSE = [
    # Country/Region ETFs for regional growth/geopolitics
    'EWJ',   # Japan
    'EWG',   # Germany  
    'EWU',   # United Kingdom
    'EWA',   # Australia
    'EWC',   # Canada
    'EWZ',   # Brazil
    'INDA',  # India
    'FXI',   # China
    'EZA',   # South Africa
    'TUR',   # Turkey
    'RSX',   # Russia
    'EWW',   # Mexico
    
    # Currency ETFs for FX/monetary policy
    'UUP',   # US Dollar Bullish
    'FXE',   # Euro
    'FXY',   # Japanese Yen
    'FXB',   # British Pound
    'FXC',   # Canadian Dollar
    'FXA',   # Australian Dollar
    'FXF',   # Swiss Franc
    'CYB',   # Chinese Yuan
    
    # Bond ETFs for rates/credit
    'TLT',   # 20+ Year Treasury
    'IEF',   # 7-10 Year Treasury
    'BND',   # Total Bond Market
    'TIP',   # TIPS (Inflation Protected)
    'LQD',   # Investment Grade Corporate
    'HYG',   # High Yield Corporate
    'EMB',   # Emerging Market Bonds
    'PCY',   # Emerging Market Sovereign
    
    # Stock Index ETFs for broad equities/growth
    'SPY',  # S&P 500
    'QQQ',   # NASDAQ 100
    'VEU',   # All-World ex-US
    'VWO',   # Emerging Markets
    'VGK',   # Europe
    'VPL',   # Asia Pacific
    'ACWI',  # All Country World Index
    
    # Commodity ETFs for inflation/supply shocks
    'GLD',   # Gold
    'SLV',   # Silver
    'USO',   # Oil
    'UNG',   # Natural Gas
    'DBC',   # Commodity Index
    'CORN',  # Corn
    'WEAT',  # Wheat
    'DBA',   # Agriculture
    'PDBC',  # Commodity Strategy
    'GSG',   # Commodity Index
]

# Federal Reserve Economic Data (FRED) indicators for macro analysis
MACRO_INDICATORS = [
    'CPIAUCSL',  # Consumer Price Index for All Urban Consumers
    'UNRATE',    # Unemployment Rate
    'FEDFUNDS',  # Federal Funds Rate
    'GDPC1',     # Real Gross Domestic Product
]

# Default trading horizon for the system
DEFAULT_HORIZON = 'months_to_years'

# Default system configuration
DEFAULT_CONFIG = {
    'max_debate_rounds': 2,
    'llm_model': 'deepseek-chat',
    'batch_size': 10,  # Number of ETFs to process in parallel
    'analysis_horizon_days': 90,  # Lookback period for analysis
    'rebalance_frequency': 'monthly',  # How often to rebalance
}

# LLM Configuration for flexibility
# Users can switch providers by editing this configuration
LLM_CONFIG = {
    'provider': 'deepseek',
    'model': 'deepseek-chat',
    'api_key_env': 'DEEPSEEK_API_KEY',
    'base_url': 'https://api.deepseek.com/v1',
    'temperature': 0.7,
    'max_tokens': 4000,
}

# Alternative LLM configurations (commented out for reference)
# To switch providers, uncomment the desired configuration and comment out the current one

# OpenAI Configuration
# LLM_CONFIG = {
#     'provider': 'openai',
#     'model': 'gpt-4',
#     'api_key_env': 'OPENAI_API_KEY',
#     'base_url': 'https://api.openai.com/v1',
#     'temperature': 0.7,
#     'max_tokens': 4000,
# }

# Anthropic Configuration
# LLM_CONFIG = {
#     'provider': 'anthropic',
#     'model': 'claude-3-sonnet-20240229',
#     'api_key_env': 'ANTHROPIC_API_KEY',
#     'base_url': 'https://api.anthropic.com',
#     'temperature': 0.7,
#     'max_tokens': 4000,
# }

# ETF Category mappings for analysis grouping
ETF_CATEGORIES = {
    'country_region': [
        'EWJ', 'EWG', 'EWU', 'EWA', 'EWC', 'EWZ', 'INDA', 'FXI', 'EZA', 'TUR', 'RSX', 'EWW'
    ],
    'currency': [
        'UUP', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA', 'FXF', 'CYB'
    ],
    'bonds': [
        'TLT', 'IEF', 'BND', 'TIP', 'LQD', 'HYG', 'EMB', 'PCY'
    ],
    'equity_indices': [
        'SPY', 'QQQ', 'VEU', 'VWO', 'VGK', 'VPL', 'ACWI'
    ],
    'commodities': [
        'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'CORN', 'WEAT', 'DBA', 'PDBC', 'GSG'
    ]
}

# Risk management parameters
RISK_CONFIG = {
    'max_position_size': 0.15,  # Maximum 15% allocation to any single ETF
    'max_category_exposure': 0.40,  # Maximum 40% allocation to any category
    'max_leverage': 1.0,  # No leverage for conservative approach
    'stop_loss_threshold': 0.10,  # 10% stop loss
    'rebalance_threshold': 0.05,  # Rebalance when allocation drifts 5%
}

# Data source configuration
DATA_CONFIG = {
    'primary_source': 'yfinance',  # Primary data source
    'backup_source': 'fredapi',   # Backup for economic data
    'cache_duration_hours': 1,     # Cache data for 1 hour
    'retry_attempts': 3,          # Number of retry attempts for failed requests
    'timeout_seconds': 30,        # Request timeout
}

# Agent configuration for the macro trading system
AGENT_CONFIG = {
    'enabled_agents': [
        'stanley_druckenmiller',
        'rakesh_jhunjhunwala',
        'risk_manager',
        'portfolio_manager'
    ],
    'agent_weights': {
        'stanley_druckenmiller': 0.40,  # Macro analysis
        'rakesh_jhunjhunwala': 0.30,    # Emerging markets
        'risk_manager': 0.20,            # Risk management
        'portfolio_manager': 0.10,       # Portfolio construction
    },
    'consensus_threshold': 0.60,  # Minimum consensus for signal execution
}

# Output configuration
OUTPUT_CONFIG = {
    'format': 'json',  # Output format for results
    'include_reasoning': True,  # Include agent reasoning in output
    'include_metrics': True,   # Include performance metrics
    'save_to_file': True,      # Save results to file
    'output_directory': 'results',  # Directory for output files
}

# Validation function to ensure configuration is valid
def validate_config():
    """Validate the configuration for consistency and completeness."""
    errors = []
    
    # Check ETF universe is not empty
    if not ETF_UNIVERSE:
        errors.append("ETF_UNIVERSE cannot be empty")
    
    # Check all ETFs in categories are in the universe
    all_categorized_etfs = set()
    for category, etfs in ETF_CATEGORIES.items():
        all_categorized_etfs.update(etfs)
    
    universe_set = set(ETF_UNIVERSE)
    if not all_categorized_etfs.issubset(universe_set):
        missing = all_categorized_etfs - universe_set
        errors.append(f"ETFs in categories but not in universe: {missing}")
    
    # Check agent weights sum to 1.0
    total_weight = sum(AGENT_CONFIG['agent_weights'].values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Agent weights sum to {total_weight}, should be 1.0")
    
    # Check risk parameters are valid
    if RISK_CONFIG['max_position_size'] > 1.0:
        errors.append("max_position_size cannot exceed 1.0")
    
    if RISK_CONFIG['max_category_exposure'] > 1.0:
        errors.append("max_category_exposure cannot exceed 1.0")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

# Initialize configuration validation on import
if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!")

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
    # --- Inflation & Prices ---
    'CPIAUCSL',      # Consumer Price Index for All Urban Consumers (Headline Inflation)
    'CPILFESL',      # Core CPI (ex Food & Energy)
    'PCEPI',         # Personal Consumption Expenditures Price Index
    'PCEPILFE',      # Core PCE Price Index (Fed's preferred inflation measure)
    'PPIFIS',        # Producer Price Index (Finished Goods)
    
    # --- Labor Market ---
    'UNRATE',        # Unemployment Rate
    'PAYEMS',        # Total Nonfarm Payrolls
    'UNEMPLOY',      # Unemployed Persons
    'CIVPART',       # Labor Force Participation Rate
    'AHETPI',        # Average Hourly Earnings (Production Workers)
    'ICSA',          # Initial Jobless Claims
    'U6RATE',        # U-6 Unemployment Rate (broader measure)
    
    # --- Growth & Output ---
    'GDPC1',         # Real Gross Domestic Product
    'GDPPOT',        # Real Potential GDP
    'INDPRO',        # Industrial Production Index
    'RSXFS',         # Advance Real Retail Sales
    'HOUST',         # Housing Starts
    'PERMIT',        # New Private Housing Permits
    
    # --- Monetary Policy & Interest Rates ---
    'FEDFUNDS',      # Federal Funds Rate
    'DGS2',          # 2-Year Treasury Constant Maturity Rate
    'DGS5',          # 5-Year Treasury Constant Maturity Rate
    'DGS10',         # 10-Year Treasury Constant Maturity Rate
    'DGS30',         # 30-Year Treasury Constant Maturity Rate
    'T10Y2Y',        # 10-Year minus 2-Year Treasury Spread (Yield Curve)
    'T10Y3M',        # 10-Year minus 3-Month Treasury Spread
    'MORTGAGE30US',  # 30-Year Fixed Rate Mortgage Average
    
    # --- Money Supply & Credit ---
    'M2SL',          # M2 Money Stock
    'TOTRESNS',      # Total Reserve Balances Maintained
    'DRTSCIS',       # Net Percentage of Banks Tightening Lending Standards
    
    # --- Trade & External ---
    'BOPGSTB',       # Trade Balance: Goods and Services
    'EXUSEU',        # US / Euro Foreign Exchange Rate
    'EXJPUS',        # Japan / US Foreign Exchange Rate
    'EXCAUS',        # Canada / US Foreign Exchange Rate
    'EXCHUS',        # China / US Foreign Exchange Rate
    'DEXUSUK',       # US / UK Foreign Exchange Rate
    
    # --- Commodities & Real Assets ---
    'DCOILWTICO',    # Crude Oil Prices: WTI
    'DCOILBRENTEU',  # Crude Oil Prices: Brent
    'GASREGW',       # US Regular All Formulations Gas Price
    
    # --- Sentiment & Leading Indicators ---
    'UMCSENT',       # University of Michigan Consumer Sentiment
    'VIXCLS',        # CBOE Volatility Index (VIX)
    'NAPM',          # ISM Manufacturing PMI (Business Activity)
    'USSLIND',       # Leading Index for the United States
    
    # --- Credit & Financial Conditions ---
    'BAMLH0A0HYM2',  # ICE BofA US High Yield Option-Adjusted Spread
    'BAA10Y',        # Moody's Seasoned Baa Corporate Bond Yield Relative to 10-Year Treasury
    'NFCI',          # Chicago Fed National Financial Conditions Index
    
    # --- Housing & Real Estate ---
    'CSUSHPISA',     # S&P/Case-Shiller U.S. National Home Price Index
    'MORTGAGE15US',  # 15-Year Fixed Rate Mortgage Average
    
    # --- Government & Fiscal ---
    'GFDEBTN',       # Federal Debt: Total Public Debt
    'FYFSD',         # Federal Surplus or Deficit
    
    # ============================================
    # INTERNATIONAL ECONOMIC INDICATORS
    # ============================================
    
    # --- CHINA ---
    'CHNCPIALLMINMEI',   # China CPI (All items)
    'CHNUNEMPMEI',       # China Unemployment Rate
    'CHNGDPNQDSMEI',     # China GDP
    'CHNGDPRAPCHG',      # China Real GDP Growth Rate
    'CHNMKTMKTDMINMEI',  # China Stock Market Index
    
    # --- EUROZONE ---
    'CP0000EZ19M086NEST',  # Euro Area HICP (Inflation)
    'LRHUTTTTEZM156S',     # Euro Area Harmonized Unemployment Rate
    'CLVMNACSCAB1GQEA19',  # Euro Area Real GDP
    'EA19GDPDEFQISMEI',    # Euro Area GDP Deflator
    'ECBDFR',              # ECB Deposit Facility Rate
    
    # --- GERMANY ---
    'DEUCPIALLMINMEI',   # Germany CPI
    'DEUUNEMPMEI',       # Germany Unemployment Rate
    'DEUPEPAC',          # Germany Industrial Production
    'DEUMKTMKTDMINMEI',  # Germany Stock Market Index
    
    # --- UNITED KINGDOM ---
    'GBRCPIALLMINMEI',   # UK CPI
    'GBRUNEMPQDSMEI',    # UK Unemployment Rate
    'GBRGDPNQDSMEI',     # UK GDP
    'BOGMBASE',          # Bank of England Base Rate
    
    # --- JAPAN ---
    'JPNCPIALLMINMEI',   # Japan CPI
    'JPNUNEMPMEI',       # Japan Unemployment Rate
    'JPNGDPNQDSMEI',     # Japan GDP
    'JPNPEPAC',          # Japan Industrial Production
    'JPNMKTMKTDMINMEI',  # Japan Stock Market Index
    
    # --- CANADA ---
    'CANCPIALLMINMEI',   # Canada CPI
    'CANUNEMPMEI',       # Canada Unemployment Rate
    'CANGDPNQDSMEI',     # Canada GDP
    'CANMKTMKTDMINMEI',  # Canada Stock Market Index
    
    # --- AUSTRALIA ---
    'AUSCPIALLQINMEI',   # Australia CPI
    'AUSUNEMPQDSMEI',    # Australia Unemployment Rate
    'AUSGDPNQDSMEI',     # Australia GDP
    
    # --- BRAZIL ---
    'BRACPIALLMINMEI',   # Brazil CPI
    'BRAUNEMPMEI',       # Brazil Unemployment Rate
    'BRAGDPNQDSMEI',     # Brazil GDP
    
    # --- INDIA ---
    'INDCPIALLMINMEI',   # India CPI
    'INDGDPNQDSMEI',     # India GDP
    
    # --- MEXICO ---
    'MEXCPIALLMINMEI',   # Mexico CPI
    'MEXUNEMPMEI',       # Mexico Unemployment Rate
    'MEXGDPNQDSMEI',     # Mexico GDP
    
    # --- SOUTH AFRICA ---
    'ZAFCPIALLMINMEI',   # South Africa CPI
    'ZAFUNEMPMEI',       # South Africa Unemployment Rate
    
    # --- SOUTH KOREA ---
    'KORCPIALLMINMEI',   # South Korea CPI
    'KORUNEMPQDSMEI',    # South Korea Unemployment Rate
    'KORGDPNQDSMEI',     # South Korea GDP
    
    # --- SWITZERLAND ---
    'CHECPIALLMINMEI',   # Switzerland CPI
    'CHEUNEMPQDSMEI',    # Switzerland Unemployment Rate
    
    # --- GLOBAL / COMPOSITE INDICATORS ---
    'OECD',              # OECD Composite Leading Indicator
    'OECD360',           # OECD Leading Indicators
    
    # --- EMERGING MARKETS ---
    'DEXBZUS',           # Brazil / US FX Rate
    'DEXINUS',           # India / US FX Rate
    'DEXMXUS',           # Mexico / US FX Rate
    'DEXSZUS',           # Switzerland / US FX Rate
    'DEXKOUS',           # South Korea / US FX Rate
    'DEXUSAL',           # Australia / US FX Rate
]

# Default trading horizon for the system
DEFAULT_HORIZON = 'months_to_years'

# Signal Agents Configuration
# These are the analyst agents that provide scores for portfolio synthesis
SIGNAL_AGENTS = [
    'MacroEconomistAgent',
    'GeopoliticalAnalystAgent'
]


# LLM Configuration for flexibility
# Users can switch providers by editing this configuration
LLM_CONFIG = {
    'provider': 'deepseek',
    'model': 'deepseek-chat',
    'api_key_env': 'DEEPSEEK_API_KEY',
    'base_url': 'https://api.deepseek.com/v1',
    'temperature': 0,  # Deterministic output
    'seed': 42,  # Fixed seed for reproducibility
    'max_tokens': 1500,  # Reduced for more focused responses
    'top_p': 1.0,  # Deterministic sampling
    'frequency_penalty': 0.0,  # No frequency penalty
    'presence_penalty': 0.0,  # No presence penalty
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

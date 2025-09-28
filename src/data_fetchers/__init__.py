"""
Data fetchers module for the global macro ETF trading system.

This module provides data fetching capabilities for:
- Economic indicators (FRED)
- ETF price data (yfinance)
- Geopolitical news (Finlight.me)
"""

from .macro_fetcher import MacroFetcher

__all__ = ['MacroFetcher']

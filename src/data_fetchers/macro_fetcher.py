"""
Macro Data Fetcher for Global Macro ETF Trading System

This module provides comprehensive data fetching capabilities for:
- Economic indicators via FRED API
- Batch ETF price/return data via yfinance
- Geopolitical and financial news via Finlight.me API

The fetcher is designed for efficient data retrieval for both individual and batch ETF processing
and is generic to handle additions via configuration list.
"""

from fredapi import Fred
import yfinance as yf
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroFetcher:
    """
    Fetches global macro data for ETF analysis including:
    - Economic indicators (FRED)
    - ETF price data (yfinance)
    - Geopolitical news (Finlight.me)
    """
    
    def __init__(self):
        """Initialize the macro data fetcher with API connections."""
        try:
            # Initialize FRED API
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                logger.warning("FRED_API_KEY not found in environment variables")
                self.fred = None
            else:
                self.fred = Fred(api_key=fred_api_key)
                logger.info("FRED API initialized successfully")
            
            # Initialize Finlight API key
            self.finlight_key = os.getenv('FINLIGHT_API_KEY')
            if not self.finlight_key:
                logger.warning("FINLIGHT_API_KEY not found in environment variables")
            
            # Set default date range (20+ years to capture major market cycles)
            # This includes: 2000 dot-com crash, 2008 financial crisis, COVID-19, etc.
            self.default_start_date = '2000-01-01'  # Extended to 20+ years
            self.default_end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info("MacroFetcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MacroFetcher: {e}")
            raise
    
    def fetch_macro_data(self, indicators: List[str]) -> Dict[str, Dict]:
        """
        Fetch economic indicators from FRED API.
        
        Args:
            indicators: List of FRED indicator codes
            
        Returns:
            Dictionary with indicator data (last 12 periods)
        """
        if not self.fred:
            logger.error("FRED API not initialized")
            return {}
        
        macro_data = {}
        
        for indicator in indicators:
            try:
                logger.info(f"Fetching FRED data for indicator: {indicator}")
                series = self.fred.get_series(indicator)
                
                if series is not None and not series.empty:
                    # Get last 12 periods of data
                    recent_data = series.tail(12)
                    macro_data[indicator] = {
                        'data': recent_data.to_dict(),
                        'latest_value': recent_data.iloc[-1] if len(recent_data) > 0 else None,
                        'periods': len(recent_data),
                        'indicator_name': indicator
                    }
                    logger.info(f"Successfully fetched {len(recent_data)} periods for {indicator}")
                else:
                    logger.warning(f"No data available for indicator: {indicator}")
                    macro_data[indicator] = {
                        'data': {},
                        'latest_value': None,
                        'periods': 0,
                        'indicator_name': indicator,
                        'error': 'No data available'
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching FRED data for {indicator}: {e}")
                macro_data[indicator] = {
                    'data': {},
                    'latest_value': None,
                    'periods': 0,
                    'indicator_name': indicator,
                    'error': str(e)
                }
        
        return macro_data
    
    def fetch_etf_data(self, etfs: List[str], start: str = None, end: str = None) -> pd.DataFrame:
        """
        Fetch batch ETF price data using yfinance.
        
        Args:
            etfs: List of ETF tickers
            start: Start date (default: 2020-01-01)
            end: End date (default: today)
            
        Returns:
            DataFrame with ETF price data
        """
        if not etfs:
            logger.warning("No ETFs provided for data fetching")
            return pd.DataFrame()
        
        start_date = start or self.default_start_date
        end_date = end or self.default_end_date
        
        try:
            logger.info(f"Fetching ETF data for {len(etfs)} ETFs from {start_date} to {end_date}")
            
            # Download data for all ETFs simultaneously
            etf_data = yf.download(
                etfs, 
                start=start_date, 
                end=end_date,
                group_by='ticker',
                progress=False
            )
            
            if etf_data.empty:
                logger.warning("No ETF data retrieved")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched data for {len(etfs)} ETFs")
            return etf_data
            
        except Exception as e:
            logger.error(f"Error fetching ETF data: {e}")
            return pd.DataFrame()
    
    def fetch_etf_returns(self, etfs: List[str], start: str = None, end: str = None) -> pd.DataFrame:
        """
        Fetch ETF returns (percentage changes) for analysis.
        
        Args:
            etfs: List of ETF tickers
            start: Start date
            end: End date
            
        Returns:
            DataFrame with ETF returns
        """
        etf_data = self.fetch_etf_data(etfs, start, end)
        
        if etf_data.empty:
            return pd.DataFrame()
        
        try:
            # Calculate returns for each ETF
            returns_data = {}
            
            for etf in etfs:
                if etf in etf_data.columns.get_level_values(0):
                    etf_close = etf_data[etf]['Close'] if isinstance(etf_data.columns, pd.MultiIndex) else etf_data[etf]
                    if not etf_close.empty:
                        returns_data[etf] = etf_close.pct_change().dropna()
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                logger.info(f"Calculated returns for {len(returns_df.columns)} ETFs")
                return returns_df
            else:
                logger.warning("No returns data calculated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating ETF returns: {e}")
            return pd.DataFrame()
    
    def fetch_geopolitical_news(self, query: str = 'geopolitical events macro impact', 
                               days_back: int = 30, max_articles: int = 100) -> List[Dict]:
        """
        Fetch geopolitical and financial news from Finlight.me API across multiple pages.
        
        Args:
            query: Search query for news articles
            days_back: Number of days to look back for articles (default: 30)
            max_articles: Maximum number of articles to fetch (default: 100)
            
        Returns:
            List of news articles with sentiment and metadata
        """
        if not self.finlight_key:
            logger.error("Finlight API key not available")
            return []
        
        url = 'https://api.finlight.me/v2/articles'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'X-API-KEY': self.finlight_key
        }
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        page = 1
        page_size = 20  # Max per page for API
        
        try:
            logger.info(f"Fetching geopolitical news for query: {query}")
            logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            while len(all_articles) < max_articles:
                body = {
                    'query': query,
                    'language': 'en',
                    'pageSize': page_size,
                    'page': page,
                    'startDate': start_date.strftime('%Y-%m-%d'),
                    'endDate': end_date.strftime('%Y-%m-%d')
                }
                
                response = requests.post(url, headers=headers, json=body, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                if not articles:
                    # No more articles available
                    break
                
                all_articles.extend(articles)
                logger.info(f"Retrieved page {page}: {len(articles)} articles (total: {len(all_articles)})")
                
                # Check if we have enough articles or if this was the last page
                if len(articles) < page_size or len(all_articles) >= max_articles:
                    break
                
                page += 1
            
            # Trim to max_articles if we got more
            all_articles = all_articles[:max_articles]
            
            logger.info(f"Total articles retrieved: {len(all_articles)} from Finlight.me")
            return all_articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from Finlight.me: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news: {e}")
            return []
    
    def fetch_comprehensive_geopolitical_news(self, days_back: int = 30) -> List[Dict]:
        """
        Fetch comprehensive geopolitical news from multiple query categories.
        
        Args:
            days_back: Number of days to look back for articles
            
        Returns:
            Deduplicated list of news articles with sentiment and metadata
        """
        queries = [
            'global macro trends geopolitical events',
            'central bank monetary policy interest rates',
            'trade war tariffs international trade',
            'geopolitical risk conflict war',
            'emerging markets crisis',
            'china economy us relations',
            'europe energy crisis recession',
            'inflation federal reserve ECB',
            'oil prices commodity markets',
            'currency crisis exchange rates'
        ]
        
        all_articles = []
        seen_identifiers = set()
        
        for query in queries:
            articles = self.fetch_geopolitical_news(query, days_back=days_back, max_articles=20)
            
            # Deduplicate by URL, title, or article ID
            for article in articles:
                # Try multiple identifiers for deduplication
                url = article.get('url', '')
                title = article.get('title', '')
                article_id = article.get('id', '')
                
                # Create a unique identifier
                identifier = url or article_id or title
                
                if identifier and identifier not in seen_identifiers:
                    seen_identifiers.add(identifier)
                    all_articles.append(article)
                elif not identifier:
                    # If no identifier, just add it (unlikely to be duplicate)
                    all_articles.append(article)
        
        logger.info(f"Total unique articles after deduplication: {len(all_articles)}")
        logger.info(f"Articles by category: {len(queries)} queries processed")
        return all_articles
    
    def fetch_comprehensive_data(self, etfs: List[str], indicators: List[str], 
                               news_queries: List[str] = None) -> Dict:
        """
        Fetch comprehensive macro data for all sources.
        
        Args:
            etfs: List of ETF tickers
            indicators: List of FRED indicators
            news_queries: List of news search queries
            
        Returns:
            Dictionary with all fetched data
        """
        logger.info("Starting comprehensive data fetch")
        
        comprehensive_data = {
            'timestamp': datetime.now().isoformat(),
            'etfs': etfs,
            'indicators': indicators,
            'macro_data': {},
            'etf_data': pd.DataFrame(),
            'etf_returns': pd.DataFrame(),
            'news_data': [],
            'errors': []
        }
        
        # Fetch macro indicators
        if indicators:
            logger.info(f"Fetching macro indicators: {indicators}")
            macro_data = self.fetch_macro_data(indicators)
            comprehensive_data['macro_data'] = macro_data
        
        # Fetch ETF data
        if etfs:
            logger.info(f"Fetching ETF data for: {etfs}")
            etf_data = self.fetch_etf_data(etfs)
            etf_returns = self.fetch_etf_returns(etfs)
            
            comprehensive_data['etf_data'] = etf_data
            comprehensive_data['etf_returns'] = etf_returns
        
        # Fetch news data
        if news_queries:
            logger.info(f"Fetching news for queries: {news_queries}")
            all_news = []
            for query in news_queries:
                news = self.fetch_geopolitical_news(query)
                all_news.extend(news)
            comprehensive_data['news_data'] = all_news
        
        logger.info("Comprehensive data fetch completed")
        return comprehensive_data
    
    def get_data_summary(self, data: Dict) -> Dict:
        """
        Generate a summary of the fetched data.
        
        Args:
            data: Comprehensive data dictionary
            
        Returns:
            Summary statistics
        """
        summary = {
            'fetch_timestamp': data.get('timestamp'),
            'etfs_count': len(data.get('etfs', [])),
            'indicators_count': len(data.get('indicators', [])),
            'macro_indicators': list(data.get('macro_data', {}).keys()),
            'etf_data_shape': data.get('etf_data', pd.DataFrame()).shape,
            'etf_returns_shape': data.get('etf_returns', pd.DataFrame()).shape,
            'news_articles_count': len(data.get('news_data', [])),
            'errors_count': len(data.get('errors', []))
        }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = MacroFetcher()
    
    # Example ETFs and indicators
    sample_etfs = ['SPY', 'QQQ', 'TLT', 'GLD']
    sample_indicators = ['CPIAUCSL', 'UNRATE', 'FEDFUNDS']
    sample_queries = ['geopolitical events macro impact', 'central bank policy']
    
    # Fetch comprehensive data
    data = fetcher.fetch_comprehensive_data(
        etfs=sample_etfs,
        indicators=sample_indicators,
        news_queries=sample_queries
    )
    
    # Print summary
    summary = fetcher.get_data_summary(data)
    print("Data Fetch Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

#!/usr/bin/env python3
"""
Comprehensive System Debug Script

This script performs a thorough analysis of the Global Macro ETF Trading System
to identify errors, stock remnants, inconsistencies, and ensure proper functionality.
"""

import os
import sys
import re
import ast
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
import pandas as pd
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemDebugger:
    """Comprehensive system debugging and validation."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.errors = []
        self.warnings = []
        self.stock_remnants = []
        self.inconsistencies = []
        self.unused_files = []
        
        # Stock-related terms to search for
        self.stock_terms = [
            'stock', 'ticker', 'earnings', 'P/E', 'sentiment', 'fundamentals', 
            'technicals', 'RSI', 'MACD', 'stock_analyzer', 'equity', 'shares',
            'dividend', 'market_cap', 'PE_ratio', 'PB_ratio', 'ROE', 'ROA'
        ]
        
        # Required imports for the system
        self.required_imports = [
            'pandas', 'numpy', 'fredapi', 'yfinance', 'requests', 
            'plotly', 'cvxpy', 'scipy', 'langgraph'
        ]
        
        # Core system files
        self.core_files = [
            'src/main.py',
            'src/config.py',
            'src/graph/macro_trading_graph.py',
            'src/data_fetchers/macro_fetcher.py',
            'src/agents/base_agent.py'
        ]
    
    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run comprehensive system debugging."""
        logger.info("🔍 Starting comprehensive system debugging...")
        
        results = {
            'file_analysis': self.analyze_files(),
            'import_analysis': self.analyze_imports(),
            'stock_remnants': self.find_stock_remnants(),
            'inconsistencies': self.find_inconsistencies(),
            'unused_files': self.find_unused_files(),
            'test_organization': self.organize_tests(),
            'llm_flexibility': self.test_llm_flexibility(),
            'batch_processing': self.verify_batch_processing(),
            'langgraph_flow': self.verify_langgraph_flow(),
            'finlight_integration': self.verify_finlight_integration(),
            'deepseek_integration': self.verify_deepseek_integration()
        }
        
        self.generate_report(results)
        return results
    
    def analyze_files(self) -> Dict[str, Any]:
        """Analyze all Python files for syntax and structure."""
        logger.info("📁 Analyzing Python files...")
        
        python_files = list(self.repo_path.rglob("*.py"))
        file_analysis = {
            'total_files': len(python_files),
            'syntax_errors': [],
            'import_errors': [],
            'structure_issues': []
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check syntax
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    file_analysis['syntax_errors'].append({
                        'file': str(file_path),
                        'error': str(e),
                        'line': e.lineno
                    })
                
                # Check for common issues
                if 'import *' in content:
                    file_analysis['structure_issues'].append({
                        'file': str(file_path),
                        'issue': 'Wildcard import detected',
                        'severity': 'warning'
                    })
                
            except Exception as e:
                file_analysis['import_errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return file_analysis
    
    def analyze_imports(self) -> Dict[str, Any]:
        """Analyze import statements and dependencies."""
        logger.info("📦 Analyzing imports and dependencies...")
        
        import_analysis = {
            'missing_imports': [],
            'unused_imports': [],
            'dependency_issues': []
        }
        
        # Check if all required imports are available
        for import_name in self.required_imports:
            try:
                __import__(import_name)
            except ImportError:
                import_analysis['missing_imports'].append(import_name)
        
        # Check pyproject.toml for dependencies
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                content = f.read()
                for import_name in self.required_imports:
                    if import_name not in content:
                        import_analysis['dependency_issues'].append({
                            'dependency': import_name,
                            'issue': 'Not found in pyproject.toml'
                        })
        
        return import_analysis
    
    def find_stock_remnants(self) -> List[Dict[str, Any]]:
        """Find any remaining stock-related code."""
        logger.info("🔍 Searching for stock remnants...")
        
        remnants = []
        python_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for term in self.stock_terms:
                    if re.search(rf'\b{term}\b', content, re.IGNORECASE):
                        # Find line numbers
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if re.search(rf'\b{term}\b', line, re.IGNORECASE):
                                remnants.append({
                                    'file': str(file_path),
                                    'term': term,
                                    'line': i,
                                    'content': line.strip()
                                })
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return remnants
    
    def find_inconsistencies(self) -> List[Dict[str, Any]]:
        """Find inconsistencies in the codebase."""
        logger.info("🔍 Looking for inconsistencies...")
        
        inconsistencies = []
        
        # Check for inconsistent naming
        python_files = list(self.repo_path.rglob("*.py"))
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for inconsistent variable naming
                if 'ETF' in content and 'etf' in content:
                    inconsistencies.append({
                        'file': str(file_path),
                        'issue': 'Inconsistent ETF/etf naming',
                        'severity': 'warning'
                    })
                
                # Check for hardcoded values that should be configurable
                hardcoded_patterns = [
                    r'["\']SPY["\']', r'["\']QQQ["\']', r'["\']TLT["\']',
                    r'["\']GLD["\']', r'["\']https://api\.deepseek\.com["\']'
                ]
                
                for pattern in hardcoded_patterns:
                    if re.search(pattern, content):
                        inconsistencies.append({
                            'file': str(file_path),
                            'issue': f'Hardcoded value found: {pattern}',
                            'severity': 'info'
                        })
                
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
        
        return inconsistencies
    
    def find_unused_files(self) -> List[str]:
        """Find potentially unused files."""
        logger.info("🗑️ Looking for unused files...")
        
        unused_files = []
        
        # Check for old agent files that might be unused
        old_agent_files = [
            'src/agents/warren_buffett.py',
            'src/agents/ben_graham.py',
            'src/agents/bill_ackman.py',
            'src/agents/cathie_wood.py',
            'src/agents/charlie_munger.py',
            'src/agents/michael_burry.py',
            'src/agents/mohnish_pabrai.py',
            'src/agents/peter_lynch.py',
            'src/agents/phil_fisher.py',
            'src/agents/aswath_damodaran.py',
            'src/agents/valuation.py',
            'src/agents/sentiment.py',
            'src/agents/fundamentals.py',
            'src/agents/technicals.py'
        ]
        
        for file_path in old_agent_files:
            full_path = self.repo_path / file_path
            if full_path.exists():
                unused_files.append(str(file_path))
        
        return unused_files
    
    def organize_tests(self) -> Dict[str, Any]:
        """Organize tests in a central location."""
        logger.info("🧪 Organizing tests...")
        
        test_organization = {
            'current_tests': [],
            'missing_tests': [],
            'test_structure': {}
        }
        
        # Find all test files
        test_files = list(self.repo_path.rglob("test_*.py"))
        for test_file in test_files:
            test_organization['current_tests'].append(str(test_file))
        
        # Check for missing tests for core components
        core_components = [
            'src/agents/macro_economist.py',
            'src/agents/geopolitical_analyst.py',
            'src/agents/correlation_specialist.py',
            'src/agents/trader_agent.py',
            'src/agents/risk_manager.py',
            'src/agents/portfolio_optimizer.py',
            'src/graph/macro_trading_graph.py',
            'src/data_fetchers/macro_fetcher.py'
        ]
        
        for component in core_components:
            component_path = Path(component)
            test_file_name = f"test_{component_path.stem}.py"
            test_path = self.repo_path / "tests" / test_file_name
            
            if not test_path.exists():
                test_organization['missing_tests'].append(component)
        
        return test_organization
    
    def test_llm_flexibility(self) -> Dict[str, Any]:
        """Test LLM provider flexibility."""
        logger.info("🤖 Testing LLM flexibility...")
        
        llm_test = {
            'config_switches': [],
            'provider_support': [],
            'flexibility_score': 0
        }
        
        # Check config.py for LLM configuration
        config_path = self.repo_path / "src" / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for multiple provider support
            providers = ['deepseek', 'openai', 'anthropic', 'groq']
            supported_providers = []
            
            for provider in providers:
                if provider in content.lower():
                    supported_providers.append(provider)
            
            llm_test['provider_support'] = supported_providers
            llm_test['flexibility_score'] = len(supported_providers) / len(providers) * 100
        
        return llm_test
    
    def verify_batch_processing(self) -> Dict[str, Any]:
        """Verify batch processing capabilities."""
        logger.info("📊 Verifying batch processing...")
        
        batch_verification = {
            'batch_etf_processing': False,
            'batch_data_fetching': False,
            'parallel_processing': False
        }
        
        # Check macro_fetcher.py for batch processing
        fetcher_path = self.repo_path / "src" / "data_fetchers" / "macro_fetcher.py"
        if fetcher_path.exists():
            with open(fetcher_path, 'r') as f:
                content = f.read()
            
            if 'batch' in content.lower() or 'etfs' in content:
                batch_verification['batch_etf_processing'] = True
            
            if 'yfinance' in content and 'download' in content:
                batch_verification['batch_data_fetching'] = True
        
        # Check for parallel processing
        if 'ray' in str(self.repo_path.rglob("*.py")):
            batch_verification['parallel_processing'] = True
        
        return batch_verification
    
    def verify_langgraph_flow(self) -> Dict[str, Any]:
        """Verify LangGraph workflow implementation."""
        logger.info("🔄 Verifying LangGraph flow...")
        
        flow_verification = {
            'graph_initialization': False,
            'node_definitions': False,
            'edge_definitions': False,
            'state_management': False
        }
        
        graph_path = self.repo_path / "src" / "graph" / "macro_trading_graph.py"
        if graph_path.exists():
            with open(graph_path, 'r') as f:
                content = f.read()
            
            if 'StateGraph' in content:
                flow_verification['graph_initialization'] = True
            
            if 'add_node' in content:
                flow_verification['node_definitions'] = True
            
            if 'add_edge' in content:
                flow_verification['edge_definitions'] = True
            
            if 'state' in content and 'dict' in content:
                flow_verification['state_management'] = True
        
        return flow_verification
    
    def verify_finlight_integration(self) -> Dict[str, Any]:
        """Verify Finlight.me API integration."""
        logger.info("🌐 Verifying Finlight integration...")
        
        finlight_verification = {
            'api_integration': False,
            'news_fetching': False,
            'error_handling': False
        }
        
        fetcher_path = self.repo_path / "src" / "data_fetchers" / "macro_fetcher.py"
        if fetcher_path.exists():
            with open(fetcher_path, 'r') as f:
                content = f.read()
            
            if 'finlight' in content.lower():
                finlight_verification['api_integration'] = True
            
            if 'fetch_geopolitical_news' in content:
                finlight_verification['news_fetching'] = True
            
            if 'try' in content and 'except' in content:
                finlight_verification['error_handling'] = True
        
        return finlight_verification
    
    def verify_deepseek_integration(self) -> Dict[str, Any]:
        """Verify DeepSeek LLM integration."""
        logger.info("🧠 Verifying DeepSeek integration...")
        
        deepseek_verification = {
            'config_present': False,
            'api_integration': False,
            'agent_usage': False
        }
        
        # Check config.py
        config_path = self.repo_path / "src" / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            if 'deepseek' in content.lower():
                deepseek_verification['config_present'] = True
        
        # Check base_agent.py
        agent_path = self.repo_path / "src" / "agents" / "base_agent.py"
        if agent_path.exists():
            with open(agent_path, 'r') as f:
                content = f.read()
            
            if 'deepseek' in content.lower():
                deepseek_verification['api_integration'] = True
        
        # Check if agents use the base agent
        agent_files = list((self.repo_path / "src" / "agents").rglob("*.py"))
        for agent_file in agent_files:
            if agent_file.name != "base_agent.py":
                with open(agent_file, 'r') as f:
                    content = f.read()
                if 'BaseAgent' in content:
                    deepseek_verification['agent_usage'] = True
                    break
        
        return deepseek_verification
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive debug report."""
        logger.info("📋 Generating debug report...")
        
        report_path = self.repo_path / "DEBUG_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# System Debug Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            # File Analysis
            f.write("## 📁 File Analysis\n\n")
            file_analysis = results['file_analysis']
            f.write(f"- Total Python files: {file_analysis['total_files']}\n")
            f.write(f"- Syntax errors: {len(file_analysis['syntax_errors'])}\n")
            f.write(f"- Import errors: {len(file_analysis['import_errors'])}\n")
            f.write(f"- Structure issues: {len(file_analysis['structure_issues'])}\n\n")
            
            # Stock Remnants
            f.write("## 🔍 Stock Remnants\n\n")
            stock_remnants = results['stock_remnants']
            if stock_remnants:
                f.write(f"Found {len(stock_remnants)} potential stock remnants:\n\n")
                for remnant in stock_remnants[:10]:  # Show first 10
                    f.write(f"- **{remnant['file']}** (line {remnant['line']}): {remnant['term']}\n")
                if len(stock_remnants) > 10:
                    f.write(f"- ... and {len(stock_remnants) - 10} more\n")
            else:
                f.write("✅ No stock remnants found\n")
            f.write("\n")
            
            # Inconsistencies
            f.write("## ⚠️ Inconsistencies\n\n")
            inconsistencies = results['inconsistencies']
            if inconsistencies:
                f.write(f"Found {len(inconsistencies)} inconsistencies:\n\n")
                for inconsistency in inconsistencies:
                    f.write(f"- **{inconsistency['file']}**: {inconsistency['issue']}\n")
            else:
                f.write("✅ No major inconsistencies found\n")
            f.write("\n")
            
            # Unused Files
            f.write("## 🗑️ Unused Files\n\n")
            unused_files = results['unused_files']
            if unused_files:
                f.write("Potentially unused files:\n\n")
                for file_path in unused_files:
                    f.write(f"- {file_path}\n")
            else:
                f.write("✅ No unused files detected\n")
            f.write("\n")
            
            # Test Organization
            f.write("## 🧪 Test Organization\n\n")
            test_org = results['test_organization']
            f.write(f"- Current test files: {len(test_org['current_tests'])}\n")
            f.write(f"- Missing tests: {len(test_org['missing_tests'])}\n")
            if test_org['missing_tests']:
                f.write("\nMissing tests for:\n")
                for component in test_org['missing_tests']:
                    f.write(f"- {component}\n")
            f.write("\n")
            
            # LLM Flexibility
            f.write("## 🤖 LLM Flexibility\n\n")
            llm_test = results['llm_flexibility']
            f.write(f"- Supported providers: {', '.join(llm_test['provider_support'])}\n")
            f.write(f"- Flexibility score: {llm_test['flexibility_score']:.1f}%\n")
            f.write("\n")
            
            # Batch Processing
            f.write("## 📊 Batch Processing\n\n")
            batch_verification = results['batch_processing']
            for key, value in batch_verification.items():
                status = "✅" if value else "❌"
                f.write(f"- {key}: {status}\n")
            f.write("\n")
            
            # LangGraph Flow
            f.write("## 🔄 LangGraph Flow\n\n")
            flow_verification = results['langgraph_flow']
            for key, value in flow_verification.items():
                status = "✅" if value else "❌"
                f.write(f"- {key}: {status}\n")
            f.write("\n")
            
            # Finlight Integration
            f.write("## 🌐 Finlight Integration\n\n")
            finlight_verification = results['finlight_integration']
            for key, value in finlight_verification.items():
                status = "✅" if value else "❌"
                f.write(f"- {key}: {status}\n")
            f.write("\n")
            
            # DeepSeek Integration
            f.write("## 🧠 DeepSeek Integration\n\n")
            deepseek_verification = results['deepseek_integration']
            for key, value in deepseek_verification.items():
                status = "✅" if value else "❌"
                f.write(f"- {key}: {status}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            f.write("1. **Clean up unused files** if confirmed safe to delete\n")
            f.write("2. **Add missing tests** for core components\n")
            f.write("3. **Fix any syntax errors** found in file analysis\n")
            f.write("4. **Verify API keys** are properly configured\n")
            f.write("5. **Test the complete workflow** end-to-end\n")
        
        logger.info(f"📋 Debug report generated: {report_path}")

def main():
    """Main function to run system debugging."""
    print("🔍 Global Macro ETF Trading System - Comprehensive Debug")
    print("=" * 60)
    
    debugger = SystemDebugger()
    results = debugger.run_comprehensive_debug()
    
    print("\n✅ Debug analysis completed!")
    print(f"📋 Report saved to: DEBUG_REPORT.md")
    
    # Summary
    print("\n📊 Summary:")
    print(f"- Files analyzed: {results['file_analysis']['total_files']}")
    print(f"- Stock remnants: {len(results['stock_remnants'])}")
    print(f"- Inconsistencies: {len(results['inconsistencies'])}")
    print(f"- Unused files: {len(results['unused_files'])}")
    print(f"- Missing tests: {len(results['test_organization']['missing_tests'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

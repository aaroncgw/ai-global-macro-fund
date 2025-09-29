#!/usr/bin/env python3
"""
Comprehensive Test Runner for Global Macro ETF Trading System

This script runs all tests in the centralized test directory and provides
detailed reporting on test coverage and results.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

class TestRunner:
    """Comprehensive test runner for the system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.test_dir = self.repo_path / "tests"
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'coverage': {}
        }
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = False) -> dict:
        """Run all tests in the system."""
        print("🧪 Running Global Macro ETF Trading System Tests")
        print("=" * 60)
        
        # Run tests by category
        test_categories = [
            ("agents", "Agent Tests"),
            ("data_fetchers", "Data Fetcher Tests"),
            ("graph", "Graph Workflow Tests"),
            ("integration", "Integration Tests"),
            ("backtesting", "Backtesting Tests")
        ]
        
        for category, description in test_categories:
            category_path = self.test_dir / category
            if category_path.exists():
                print(f"\n📋 {description}")
                print("-" * 40)
                self.run_category_tests(category, verbose, coverage)
            else:
                print(f"⚠️  No tests found for {category}")
        
        # Generate summary
        self.generate_summary()
        return self.results
    
    def run_category_tests(self, category: str, verbose: bool, coverage: bool) -> None:
        """Run tests for a specific category."""
        category_path = self.test_dir / category
        
        # Find all test files in the category
        test_files = list(category_path.rglob("test_*.py"))
        
        if not test_files:
            print(f"   No test files found in {category}")
            return
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
        
        # Add test files
        cmd.extend([str(f) for f in test_files])
        
        # Add test directory to path
        cmd.extend(["--tb=short", "-x"])  # Stop on first failure
        
        try:
            print(f"   Running {len(test_files)} test files...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            # Parse results
            self.parse_pytest_output(result.stdout, result.stderr, result.returncode)
            
        except Exception as e:
            print(f"   ❌ Error running tests: {e}")
            self.results['errors'].append(f"Category {category}: {e}")
    
    def parse_pytest_output(self, stdout: str, stderr: str, returncode: int) -> None:
        """Parse pytest output to extract test results."""
        lines = stdout.split('\n')
        
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Extract numbers from line like "5 passed, 2 failed in 0.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        self.results['passed'] += int(parts[i-1])
                    elif part == 'failed':
                        self.results['failed'] += int(parts[i-1])
                    elif part == 'skipped':
                        self.results['skipped'] += int(parts[i-1])
        
        if returncode != 0:
            self.results['errors'].append(f"Tests failed with return code {returncode}")
            if stderr:
                self.results['errors'].append(f"Error output: {stderr}")
    
    def generate_summary(self) -> None:
        """Generate test summary."""
        self.results['total_tests'] = (
            self.results['passed'] + 
            self.results['failed'] + 
            self.results['skipped']
        )
        
        success_rate = (
            self.results['passed'] / self.results['total_tests'] * 100 
            if self.results['total_tests'] > 0 else 0
        )
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"⏭️  Skipped: {self.results['skipped']}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n⚠️  Errors: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"   - {error}")
        
        # Save results to file
        results_file = self.repo_path / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📋 Detailed results saved to: {results_file}")
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> None:
        """Run a specific test file."""
        print(f"🧪 Running specific test: {test_path}")
        
        cmd = ["python", "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"❌ Error running test: {e}")

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run Global Macro ETF Trading System Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--category", help="Run tests for specific category (agents, graph, etc.)")
    parser.add_argument("--test", help="Run specific test file")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.test:
        runner.run_specific_test(args.test, args.verbose)
    elif args.category:
        runner.run_category_tests(args.category, args.verbose, args.coverage)
    else:
        runner.run_all_tests(args.verbose, args.coverage)
    
    return 0 if runner.results['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

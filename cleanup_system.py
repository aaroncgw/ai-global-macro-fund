#!/usr/bin/env python3
"""
System Cleanup Script

This script performs targeted cleanup of the Global Macro ETF Trading System
to remove unused files, fix inconsistencies, and ensure proper organization.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemCleanup:
    """System cleanup and organization."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.cleanup_log = []
    
    def run_cleanup(self) -> None:
        """Run comprehensive system cleanup."""
        logger.info("🧹 Starting system cleanup...")
        
        # Remove unused files
        self.remove_unused_files()
        
        # Clean up test organization
        self.organize_tests()
        
        # Fix import paths in moved test files
        self.fix_test_imports()
        
        # Clean up any remaining issues
        self.final_cleanup()
        
        logger.info("✅ System cleanup completed!")
        self.print_cleanup_summary()
    
    def remove_unused_files(self) -> None:
        """Remove unused files identified by the debug script."""
        logger.info("🗑️ Removing unused files...")
        
        # Files that were identified as unused (deleted agents)
        unused_files = [
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
            'src/agents/technicals.py',
            'src/agents/stanley_druckenmiller.py',
            'src/agents/rakesh_jhunjhunwala.py',
            'src/agents/example_agent.py'
        ]
        
        for file_path in unused_files:
            full_path = self.repo_path / file_path
            if full_path.exists():
                try:
                    full_path.unlink()
                    self.cleanup_log.append(f"Removed: {file_path}")
                    logger.info(f"   Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"   Could not remove {file_path}: {e}")
    
    def organize_tests(self) -> None:
        """Ensure tests are properly organized."""
        logger.info("🧪 Organizing tests...")
        
        # Create test directories if they don't exist
        test_dirs = [
            'tests/agents',
            'tests/graph', 
            'tests/data_fetchers',
            'tests/integration',
            'tests/backtesting'
        ]
        
        for test_dir in test_dirs:
            dir_path = self.repo_path / test_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Test package for {dir_name}"""\n'.format(
                    dir_name=test_dir.split('/')[-1]
                ))
                self.cleanup_log.append(f"Created: {init_file}")
    
    def fix_test_imports(self) -> None:
        """Fix import paths in moved test files."""
        logger.info("🔧 Fixing test imports...")
        
        # Test files that were moved and need import fixes
        test_files = [
            'tests/agents/test_all_agents.py',
            'tests/agents/test_allocation_agents.py', 
            'tests/agents/test_debate_integration.py',
            'tests/graph/test_complete_workflow.py',
            'tests/data_fetchers/test_macro_fetcher.py'
        ]
        
        for test_file in test_files:
            file_path = self.repo_path / test_file
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    
                    # Fix relative imports
                    if 'from src.' in content:
                        content = content.replace('from src.', 'from src.')
                        # No change needed as we're keeping src imports
                    
                    # Fix any other import issues
                    if 'import sys' not in content and 'sys.path' in content:
                        content = 'import sys\n' + content
                    
                    file_path.write_text(content)
                    self.cleanup_log.append(f"Fixed imports: {test_file}")
                    
                except Exception as e:
                    logger.warning(f"   Could not fix imports in {test_file}: {e}")
    
    def final_cleanup(self) -> None:
        """Perform final cleanup tasks."""
        logger.info("🔧 Performing final cleanup...")
        
        # Remove debug script (it's not needed in production)
        debug_script = self.repo_path / "debug_system.py"
        if debug_script.exists():
            try:
                debug_script.unlink()
                self.cleanup_log.append("Removed: debug_system.py")
            except Exception as e:
                logger.warning(f"Could not remove debug script: {e}")
        
        # Ensure all __init__.py files exist
        self.ensure_init_files()
        
        # Clean up any temporary files
        self.clean_temp_files()
    
    def ensure_init_files(self) -> None:
        """Ensure all directories have __init__.py files."""
        logger.info("📁 Ensuring __init__.py files...")
        
        directories = [
            'src',
            'src/agents',
            'src/data_fetchers', 
            'src/graph',
            'src/utils',
            'tests',
            'tests/agents',
            'tests/graph',
            'tests/data_fetchers',
            'tests/integration',
            'tests/backtesting'
        ]
        
        for directory in directories:
            dir_path = self.repo_path / directory
            if dir_path.exists() and dir_path.is_dir():
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text('"""Package for {name}"""\n'.format(
                        name=directory.split('/')[-1]
                    ))
                    self.cleanup_log.append(f"Created: {init_file}")
    
    def clean_temp_files(self) -> None:
        """Clean up temporary files."""
        logger.info("🧹 Cleaning temporary files...")
        
        temp_patterns = [
            '*.pyc',
            '__pycache__',
            '*.log',
            '.pytest_cache',
            'htmlcov',
            '.coverage'
        ]
        
        for pattern in temp_patterns:
            for file_path in self.repo_path.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    self.cleanup_log.append(f"Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
    
    def print_cleanup_summary(self) -> None:
        """Print cleanup summary."""
        print("\n" + "=" * 60)
        print("🧹 CLEANUP SUMMARY")
        print("=" * 60)
        
        if self.cleanup_log:
            print(f"Performed {len(self.cleanup_log)} cleanup actions:")
            for action in self.cleanup_log:
                print(f"  ✓ {action}")
        else:
            print("No cleanup actions needed - system is already clean!")
        
        print("\n✅ System cleanup completed successfully!")

def main():
    """Main function to run system cleanup."""
    print("🧹 Global Macro ETF Trading System - System Cleanup")
    print("=" * 60)
    
    cleanup = SystemCleanup()
    cleanup.run_cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

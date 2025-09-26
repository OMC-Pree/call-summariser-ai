#!/usr/bin/env python3
"""
Test runner for Call Summarizer project

Runs all unit tests and generates coverage reports.
"""

import unittest
import sys
import os
from pathlib import Path
import coverage

def run_tests():
    """Run all tests with coverage reporting"""

    # Initialize coverage
    cov = coverage.Coverage(source=['summariser'])
    cov.start()

    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    suite = loader.discover(test_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)
    cov.report()

    # Generate HTML report
    html_dir = test_dir / 'coverage_html'
    cov.html_report(directory=str(html_dir))
    print(f"\nHTML coverage report generated: {html_dir}/index.html")

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())
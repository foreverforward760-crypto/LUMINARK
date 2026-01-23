#!/usr/bin/env python3
"""
Standalone test of SAP Diagnostic
"""

# Direct import to avoid __init__.py dependencies
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import only what we need
from sap_yunus.sap_diagnostic import SAPDiagnostic

def main():
    diagnostic = SAPDiagnostic()

    print("Testing SAP Diagnostic with provided scores...")
    print()

    scores = {
        'complexity': 8.0,
        'stability': 5.0,
        'tension': 3.0,
        'adaptability': 9.0,
        'coherence': 7.0
    }

    report = diagnostic.generate_report(scores)
    print(report)

if __name__ == "__main__":
    main()

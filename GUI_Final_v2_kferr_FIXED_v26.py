#!/usr/bin/env python3
"""Compatibility launcher for the revised interface-independent engine."""
from critical_engine import PhysicsEngine, SKLEARN_AVAILABLE
from analysis_gui import main, run_gui_analysis

if __name__ == '__main__':
    raise SystemExit(main())

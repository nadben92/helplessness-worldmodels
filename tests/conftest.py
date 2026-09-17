"""Empty on purpose: its only job is to make tests/ itself a directory
pytest inserts onto sys.path (pytest always does this for a directory
holding a conftest.py it collects), so tests/_experiment_loader.py is
importable as a bare module from any test file anywhere under tests/,
without every test file re-deriving its own relative path to tests/.
"""

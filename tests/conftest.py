import os
import sys

# Make KataGo's pure-Python reference importable as `katago.*` for cross-checking.
_KATAGO_PYTHON = os.path.join(os.path.dirname(__file__), "..", "katago", "python")
sys.path.insert(0, os.path.abspath(_KATAGO_PYTHON))

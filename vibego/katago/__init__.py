"""Direct access to KataGo's *exported* model files (`.bin.gz`).

The rest of vibego talks to KataGo through the analysis protocol (a subprocess, see
`vibego/engine/proxy.py`). This package instead opens the weights themselves, so a released
net can be inspected, edited, and written back out as a file the stock KataGo engine loads.
"""

from .binmodel import KataModel, read_model, write_model  # noqa: F401

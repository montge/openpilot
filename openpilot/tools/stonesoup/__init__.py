"""Stone Soup comparison tools (requires the optional 'stonesoup' pip package).

The adapter import is lazy so the package (and test collection) works
without the optional dependency installed.
"""

__all__ = ['OpenpilotAdapter']


def __getattr__(name):
  if name == 'OpenpilotAdapter':
    from openpilot.tools.stonesoup.adapters import OpenpilotAdapter
    return OpenpilotAdapter
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

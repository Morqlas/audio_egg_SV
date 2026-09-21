"""Make stdout/stderr UTF-8 safe.

Several analysis scripts print arrows, deltas and minus signs. On a Windows
console defaulting to cp1252 those raise UnicodeEncodeError *after* the
results have already been written to disk. Importing this module makes the
streams tolerant. It affects console output only -- no computed value changes.
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream (e.g. piped/captured)
        pass

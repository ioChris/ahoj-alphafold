# -*- coding: utf-8 -*-
"""pytest set-up: the modules in this folder import each other by top-level name (they run as
scripts from here), so put this folder on sys.path whatever directory pytest is started from."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

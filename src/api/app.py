"""Public FastAPI app module.

The implementation lives in `src.api.app_impl`. This module keeps the stable
`src.api.app:app` import path used by Uvicorn, tests, and scripts.
"""

from __future__ import annotations

import sys

from src.api import app_impl as _app_impl

sys.modules[__name__] = _app_impl

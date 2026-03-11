import os
import sys

# Ensure project root is on sys.path so we can import src.webapp.app
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.webapp.app import app  # noqa: E402

# Vercel looks for "app" in Python entrypoints (WSGI).


import sys
import os

# Add project root to path so app.py can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel expects the WSGI callable to be named 'app'

"""
run_web.py
──────────
Single command launcher for the Zomato AI Concierge Web Application.
Starts the high-performance ASGI server on port 8000.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.server import start_server

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    print("=" * 65)
    print(f"🍽️  ZOMATO AI CONCIERGE — HYPERLOCAL SENSORY DISCOVERY")
    print("=" * 65)
    print(f"🚀 Running at: http://{host}:{port}")
    print(f"✨ Design: Stitch 'Hyperlocal Sensory AI' (Dark Glassmorphic)")
    print(f"⚡ LLM Provider: {os.getenv('LLM_PROVIDER', 'groq').upper()}")
    print("=" * 65)
    start_server(host=host, port=port)

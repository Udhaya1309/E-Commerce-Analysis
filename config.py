"""
Configuration for AI Marketing Agent.
API key is loaded from environment variable OPENAI_API_KEY for security.
"""
import os
from pathlib import Path

# Load from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Ecommerce_Consumer_Behavior_Analysis_Data.csv"

# OpenAI (use env variable - never commit actual key)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Probability thresholds for strategy (model-driven, not rule-based)
HIGH_PROB_THRESHOLD = 0.6   # Upsell
LOW_PROB_THRESHOLD = 0.35  # Below = Re-engagement; between = Nurture

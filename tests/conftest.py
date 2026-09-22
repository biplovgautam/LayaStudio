import sys
from pathlib import Path

# Run the tests against the checkout, not an installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

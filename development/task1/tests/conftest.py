import sys
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("VERA_RUNTIME_PATH", str(ROOT / "task1-retrieval")))
sys.path.insert(0, str(ROOT / "development/task1/scripts"))

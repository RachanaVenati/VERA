"""Development entry point for the canonical standalone Jina package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'task1-jina'))
from jina_retrieval import main

if __name__ == '__main__':
    main()

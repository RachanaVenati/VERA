"""Shared persistent store implementation from the Jina submission package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'task1-jina'))
from jina_vector_store import VectorWriter, search

"""
Run this ONCE to load WHO medical guidelines into Qdrant.
From E:\ORIAN\backend:
    python seed_knowledge.py
"""
import sys, os
from pathlib import Path

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from backend.knowledge import seed_knowledge

if __name__ == "__main__":
    print("Seeding ORIANMed medical knowledge base...")
    seed_knowledge()
    print("Done! Qdrant is ready.")
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json, uuid

COLLECTION = "medical_knowledge"
MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs on CPU, ~80MB

# Lazy globals — loaded once on first use
_client  = None
_model   = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        key = os.getenv("QDRANT_API_KEY", "").strip()
        if url and key:
            _client = QdrantClient(url=url, api_key=key)
            print(f"[Qdrant] connected to cloud: {url[:40]}...")
        else:
            # Fallback: in-memory Qdrant (no persistence, good for testing)
            _client = QdrantClient(":memory:")
            print("[Qdrant] WARNING — running in-memory. Add QDRANT_URL + QDRANT_API_KEY to .env")
    return _client


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[Qdrant] loading sentence-transformer model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[Qdrant] model ready")
    return _model


def ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[Qdrant] created collection: {COLLECTION}")


def seed_knowledge(data_path: str = None):
    """
    Load WHO medical guidelines into Qdrant.
    Run once: python -c "from backend.knowledge import seed_knowledge; seed_knowledge()"
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "who_guidelines.json"

    ensure_collection()
    client = get_client()
    model  = get_model()

    with open(data_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    points = []
    for item in guidelines:
        vector = model.encode(item["text"]).tolist()
        points.append(PointStruct(
            id      = str(uuid.uuid4()),
            vector  = vector,
            payload = {
                "text":      item["text"],
                "condition": item.get("condition", ""),
                "source":    item.get("source", "WHO"),
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[Qdrant] seeded {len(points)} medical guidelines")


def search_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """
    Search medical knowledge base for relevant guidelines.
    Returns top_k most relevant results.
    """
    try:
        client = get_client()
        model  = get_model()

        # Check collection exists and has data
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            return []

        vector = model.encode(query).tolist()
        results = client.search(
            collection_name = COLLECTION,
            query_vector    = vector,
            limit           = top_k,
            score_threshold = 0.4,   # only return if relevance > 40%
        )

        return [
            {
                "text":      r.payload.get("text", ""),
                "condition": r.payload.get("condition", ""),
                "score":     round(r.score, 2),
            }
            for r in results
        ]
    except Exception as e:
        print(f"[Qdrant] search error: {e}")
        return []


def format_knowledge_context(results: list[dict]) -> str:
    """Format Qdrant results into a string to inject into the AI prompt."""
    if not results:
        return ""
    lines = ["Relevant medical knowledge (WHO guidelines):"]
    for r in results:
        lines.append(f"- {r['text']}")
    return "\n".join(lines)
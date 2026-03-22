import os, uuid, json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "medical_knowledge"

_client = None
_model  = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        key = os.getenv("QDRANT_API_KEY", "").strip()
        if url and key:
            _client = QdrantClient(url=url, api_key=key)
            print(f"[Qdrant] connected to cloud: {url[:40]}...")
        else:
            _client = QdrantClient(":memory:")
            print("[Qdrant] WARNING — running in-memory. Add QDRANT_URL + QDRANT_API_KEY to .env")
    return _client


def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        print("[Qdrant] loading fastembed model...")
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("[Qdrant] fastembed model ready")
    return _model


def embed(text: str) -> list[float]:
    model = get_model()
    vectors = list(model.embed([text]))
    return vectors[0].tolist()


def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[Qdrant] created collection: {COLLECTION}")


def seed_knowledge(data_path: str = None):
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "who_guidelines.json"

    ensure_collection()
    client = get_client()

    with open(data_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    points = []
    for item in guidelines:
        vector = embed(item["text"])
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
    try:
        client = get_client()
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            return []

        vector = embed(query)
        results = client.search(
            collection_name = COLLECTION,
            query_vector    = vector,
            limit           = top_k,
            score_threshold = 0.4,
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
    if not results:
        return ""
    lines = ["Relevant medical knowledge (WHO guidelines):"]
    for r in results:
        lines.append(f"- {r['text']}")
    return "\n".join(lines)
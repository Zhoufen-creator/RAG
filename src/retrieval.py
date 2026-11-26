import json
from pathlib import Path
import numpy as np
from src.embedding import embed_query, MODEL_NAME

INDEX_DIR = Path("data/index")

def load_index():
    vectors = np.load(INDEX_DIR / "vectors.npy")
    meta = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    if meta["model"] != MODEL_NAME:
        raise ValueError(
            f"Index đang được tạo bằng {meta['model']} nhưng đang dùng {MODEL_NAME}"
        )
    return vectors, meta["chunk"]

def retrieval(query, vectors, chunks, top_k=5):
    query_vec = embed_query(query)
    scores = vectors @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({**chunks[idx], "score": float(scores[idx])})
    return results

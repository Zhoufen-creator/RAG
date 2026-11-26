import json
from pathlib import Path
import numpy as np
from src.chunking import chunk_document, with_context
from src.embedding import embed_passages, MODEL_NAME

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")

def main():
    records = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ")
        for i, (context, piece) in enumerate(chunk_document(text, title)):
            records.append({
                "source": path.name,
                "chunk_index": i,
                "context": context,
                "text": piece,
            })

    print(f"Tổng cộng: {len(records)} chunk")

    vectors = embed_passages([with_context(r["context"], r["text"]) for r in records])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)

    meta = {"model": MODEL_NAME, "chunk": records}
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Đã lưu vào data/index/")

if __name__ == "__main__":
    main()
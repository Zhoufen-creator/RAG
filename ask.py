from src.generate import generate_answer
from src.retrieval import load_index, retrieval

TOP_k = 3

def main():
    vectors, chunks = load_index()
    query = input("Câu hỏi:")
    results = retrieval(query, vectors, chunks)

    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank} | điểm {r['score']:.3f} | {r['source']} (chunk {r['chunk_index']})")
        print(r["text"])

if __name__ == "__main__":
    main()
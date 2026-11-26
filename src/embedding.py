from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}
QUERY_PREFIX, PASSAGE_PREFIX = PREFIXES.get(MODEL_NAME, ("", ""))

_model = None

def get_model():
    global _model
    if _model is None:
        print("Đang load model")
        _model = SentenceTransformer(MODEL_NAME)
        print("Đã load model")
    return _model

def embed_passages(texts):
    model = get_model()
    return model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )

def embed_query(query):
    model = get_model()
    return model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
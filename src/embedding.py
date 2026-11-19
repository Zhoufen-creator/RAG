from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        print("Đang load model")
        _model = SentenceTransformer(MODEL_NAME)
        print("Đã load model")     
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
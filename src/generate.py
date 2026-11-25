from torchgen.utils import context
from email import message
import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_MODEL="openai/gpt-oss-20b"

def build_promt(query, chunks):
    context = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1)
    )

    return (
        "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.\n"
        "Chỉ dùng thông tin trong phần TÀI LIỆU bên dưới. "
        'Nếu tài liệu không có thông tin, hãy trả lời đúng câu: '
        '"Tôi không tìm thấy thông tin này trong tài liệu."\n'
        "Ghi số thứ tự đoạn bạn dùng, ví dụ [1].\n\n"
        f"TÀI LIỆU:\n{context}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "TRẢ LỜI:"
    )

def call_GPT(promt):
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model" : GROQ_MODEL,
            "message" : [{"role" : "user", "content" : promt}],
            "stream" : False
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def generate_answer(query, chunks):
    promt = build_promt(query, chunks)
    return call_GPT(promt)
import os
import time
import requests
from dotenv import load_dotenv
from src.chunking import with_context

load_dotenv()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[{i}] {with_context(c.get('context', ''), c['text'])}" 
        for i, c in enumerate(chunks, start=1)
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

def call_GPT(prompt):
    api_key = os.getenv("GROQ_API_KEY")

    for attempt in range(4):
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2000,
                "reasoning_effort": "low",
            },
            timeout=300,
        )
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if not resp.ok:
            raise RuntimeError(f"Groq lỗi {resp.status_code}: {resp.text}")
        content = resp.json()["choices"][0]["message"].get("content")
        return content or "(model không trả về nội dung)"

    raise RuntimeError("Groq báo quá giới hạn nhiều lần, hãy thử lại sau.")

def generate_answer(query, chunks):
    prompt = build_prompt(query, chunks)
    return call_GPT(prompt)
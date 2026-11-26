import re

HEADING_RE = re.compile(r"^=+\s*(.+?)\s*=+$")
SEPARATORS = ["\n\n", "\n", ". ", " "]

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def split_sections(texts):
    sections = []
    heading = ""
    buffer = []
    for line in texts.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = m.group(1)
            buffer = []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    return sections

def recursive_split(text, max_chars=800, separators=SEPARATORS):
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    sep, rest = separators[0], separators[1:]
    chunks = []
    current =""
    for part in text.split(sep):
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > max_chars:
            chunks.extend(recursive_split(part, max_chars, rest))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks

def chunk_document(text, title, max_chars=800):
    results = []
    for heading, body in split_sections(text):
        context = f"{title} - {heading}" if heading else title
        for piece in recursive_split(body, max_chars):
            if piece.strip():
                results.append((context, piece.strip()))
    return results

def with_context(context, text):
    return f"{context}\n{text}" if context else text
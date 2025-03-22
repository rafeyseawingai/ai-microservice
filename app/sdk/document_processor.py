class DocumentProcessor:
    def __init__(self, text, max_tokens=80):
        self.text = text
        self.max_tokens = max_tokens
        self.chunks = self.chunk_text()

    def chunk_text(self):
        sentences = self.text.split("\n")
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= self.max_tokens:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

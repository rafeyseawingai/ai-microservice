from transformers import pipeline
from app.sdk.logger import Logger

class AnswerGenerator:
    def __init__(self):
        self.llm = pipeline("text-generation", model="google/flan-t5-xl")

    def generate_answer(self, query, context):
        if not context:
            return "No relevant information found in the document."
        prompt = f"Answer the query based only on the given context.\\n\\nContext:\\n{context}\\n\\nQuery: {query}\\nAnswer:"
        response = self.llm(prompt, max_length=100, do_sample=True)[0]["generated_text"].strip()
        Logger.log("Answer generation completed")
        return response

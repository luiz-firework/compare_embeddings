import os
import openai


class Embeddings:
    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def create(self, text):
        """Produce an embedding for the text."""
        model = "text-embedding-ada-002"
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response["data"][0]["embedding"]
        return embedding

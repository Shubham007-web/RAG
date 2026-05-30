from typing import List, Sequence

from ollama import embed


class OllamaEmbedder:
    def __init__(self, model: str):
        self.model = model
        self.max_embed_chars = 3000

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for t in texts:
            # locally truncate long texts to avoid server-side context length errors
            short = t if len(t) <= self.max_embed_chars else t[: self.max_embed_chars]
            response = embed(model=self.model, input=short, truncate=True)
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                raise RuntimeError("No embeddings returned from ollama.embed for a text chunk")
            # embeddings may be a nested sequence; normalize to a single vector
            first = embeddings[0] if len(embeddings) > 0 else embeddings
            if isinstance(first, (list, tuple)):
                vec = list(first)
            else:
                vec = list(embeddings)
            results.append(vec)
        return results

    def embed_text(self, text: str) -> List[float]:
        result = self.embed_texts([text])
        return result[0] if result else []

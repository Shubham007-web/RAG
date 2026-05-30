from typing import List, Optional

from ollama import chat


class OllamaChatModel:
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, context: Optional[List[str]] = None, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context and knowledge graph facts."},
        ]
        if context:
            for snippet in context:
                messages.append({"role": "system", "content": snippet})
        messages.append({"role": "user", "content": prompt})

        response = chat(model=self.model, messages=messages)
        return response.message.content

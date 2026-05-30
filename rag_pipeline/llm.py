import os
from typing import List, Optional

from ollama import chat

from .config import settings


def setup_langsmith():
    """Setup LangSmith tracing if enabled."""
    if settings.enable_langsmith:
        if not settings.langsmith_api_key:
            print("Warning: LangSmith enabled but LANGSMITH_API_KEY not set")
            return
        
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"] = "true"


class OllamaChatModel:
    def __init__(self, model: str):
        self.model = model
        setup_langsmith()

    def generate(self, prompt: str, context: Optional[List[str]] = None, temperature: float = 0.1, trace_name: str = "llm_generation") -> str:
        """Generate response using Ollama.
        
        Args:
            prompt: User prompt/question
            context: List of context snippets to include as system messages
            temperature: Temperature for generation (0.0-1.0)
            trace_name: Name for LangSmith trace (if enabled)
        
        Returns:
            Generated response text
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context and knowledge graph facts."},
        ]
        if context:
            for snippet in context:
                messages.append({"role": "system", "content": snippet})
        messages.append({"role": "user", "content": prompt})

        # If LangSmith is enabled, we could add tracing here
        # For now, Ollama doesn't support LangSmith callbacks directly
        # but we can log the call for observability
        if settings.enable_langsmith:
            self._log_to_langsmith(trace_name, prompt, context)

        response = chat(model=self.model, messages=messages)
        return response.message.content
    
    def _log_to_langsmith(self, trace_name: str, prompt: str, context: Optional[List[str]] = None):
        """Log LLM call to LangSmith for observability."""
        try:
            from langsmith import Client
            
            client = Client()
            # This is a simple logging mechanism
            # In production, you might want to use langsmith decorators with LangChain
            # For now we just track that a call was made
            run = client.create_run(
                name=trace_name,
                run_type="llm",
                inputs={"prompt": prompt, "context": context or []},
            )
            # In a real scenario, you'd update this with the output
            # For now, we just log the call occurred
            
        except Exception as e:
            # Silently fail if LangSmith is not properly configured
            pass

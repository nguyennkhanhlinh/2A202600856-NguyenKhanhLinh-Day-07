from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        """Combine retrieved chunks and the question into a grounded prompt."""
        if results:
            context_blocks = []
            for index, result in enumerate(results, start=1):
                source = result.get("metadata", {}).get("source", "unknown")
                context_blocks.append(
                    f"[{index}] (source: {source})\n{result['content']}"
                )
            context = "\n\n".join(context_blocks)
        else:
            context = "(no relevant context found)"

        return (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def answer(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant chunks, build a grounded prompt, and call the LLM."""
        results = self.store.search(question, top_k=top_k)
        prompt = self._build_prompt(question, results)
        return self.llm_fn(prompt)

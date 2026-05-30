from pathlib import Path
import sys

# Ensure project root is on sys.path when running this script from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.config import settings
from rag_pipeline.doc_processor import DocumentProcessor
from rag_pipeline.embeddings import OllamaEmbedder
from rag_pipeline.kg import KnowledgeGraphBuilder, NetworkXKnowledgeGraph
from rag_pipeline.retriever import HybridRetriever


def main():
    print("Building pipeline assets...")
    processor = DocumentProcessor(settings.data_dir)
    chunks = [chunk.__dict__ for chunk in processor.process_all()]
    print(f"Extracted {len(chunks)} chunks from data directory {settings.data_dir}")

    embedder = OllamaEmbedder(settings.embedding_model)
    retriever = HybridRetriever(chunks, embedder, str(settings.vector_db_dir))
    print(f"Vector store initialized under {settings.vector_db_dir}")

    graph_builder = KnowledgeGraphBuilder(NetworkXKnowledgeGraph())
    graph_builder.build_graph(chunks)
    settings.kg_json_path.parent.mkdir(parents=True, exist_ok=True)
    graph_builder.kg.save(settings.kg_json_path)
    print(f"Knowledge graph prototype saved to {settings.kg_json_path}")

    print("Pipeline build complete.")


if __name__ == "__main__":
    main()

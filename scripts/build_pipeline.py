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
    print("Building pipeline assets with incremental indexing...\n")
    
    # Use incremental processing - only reprocess files that have changed
    processor = DocumentProcessor(settings.data_dir, incremental=True)
    chunks, stats = processor.process_all(incremental=True)
    
    print(f"Processing Stats:")
    print(f"  Total files found: {stats['total_files']}")
    print(f"  Files processed: {stats['processed_files']}")
    print(f"  Files skipped (unchanged): {stats['skipped_files']}")
    print(f"  Total chunks created: {stats['total_chunks']}\n")
    
    # Convert chunks to dict format
    chunks_dicts = [chunk.__dict__ if hasattr(chunk, '__dict__') else chunk for chunk in chunks]

    embedder = OllamaEmbedder(settings.embedding_model)
    retriever = HybridRetriever(chunks_dicts, embedder, str(settings.vector_db_dir))
    print(f"Vector store initialized under {settings.vector_db_dir}")

    graph_builder = KnowledgeGraphBuilder(NetworkXKnowledgeGraph())
    graph_builder.build_graph(chunks_dicts)
    settings.kg_json_path.parent.mkdir(parents=True, exist_ok=True)
    graph_builder.kg.save(settings.kg_json_path)
    print(f"Knowledge graph saved to {settings.kg_json_path}\n")

    print("✓ Pipeline build complete!")


if __name__ == "__main__":
    main()

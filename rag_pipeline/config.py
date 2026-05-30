import os
from dataclasses import dataclass
from pathlib import Path


def default_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    data_dir: Path = default_path(os.getenv("DATA_DIR", "data"))
    vector_db_dir: Path = default_path(os.getenv("VECTOR_DB_DIR", "vector_db"))
    kg_json_path: Path = default_path(os.getenv("KG_JSON_PATH", "kg_store.json"))
    chunk_size: int = env_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = env_int("CHUNK_OVERLAP", 200)
    text_splitter_mode: str = env_str("TEXT_SPLITTER_MODE", "recursive")
    embedding_model: str = env_str("EMBEDDING_MODEL", "nomic-embed-text:latest")
    llm_model: str = env_str("LLM_MODEL", "qwen:latest")
    reranker_model: str = env_str("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    neo4j_uri: str = env_str("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = env_str("NEO4J_USER", "neo4j")
    neo4j_password: str = env_str("NEO4J_PASSWORD", "password")
    top_k: int = env_int("TOP_K", 20)
    final_k: int = env_int("FINAL_K", 5)
    bm25_weight: float = env_float("BM25_WEIGHT", 0.5)
    vector_weight: float = env_float("VECTOR_WEIGHT", 0.5)
    reranker_top_k: int = env_int("RERANKER_TOP_K", 5)
    log_level: str = env_str("LOG_LEVEL", "INFO")


settings = Settings()

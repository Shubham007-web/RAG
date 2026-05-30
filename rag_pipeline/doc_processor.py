import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from pypdf import PdfReader

from .chunking import RecursiveCharacterTextSplitter
from .config import settings
from .utils import ensure_dir, file_hash, normalize_text, now_iso


class FileHashManifest:
    """Tracks file hashes for incremental indexing."""
    
    def __init__(self, manifest_path: Path = settings.manifest_path):
        self.manifest_path = Path(manifest_path)
        self.manifest: Dict[str, str] = {}
        self.load()
    
    def load(self):
        """Load manifest from disk."""
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text())
            except Exception:
                self.manifest = {}
        else:
            self.manifest = {}
    
    def save(self):
        """Save manifest to disk."""
        ensure_dir(self.manifest_path.parent)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
    
    def get_hash(self, file_path: Path) -> Optional[str]:
        """Get stored hash for a file."""
        return self.manifest.get(str(file_path.absolute()))
    
    def update_hash(self, file_path: Path, hash_value: str):
        """Update hash for a file."""
        self.manifest[str(file_path.absolute())] = hash_value
    
    def needs_processing(self, file_path: Path) -> bool:
        """Check if file is new or has been modified."""
        current_hash = file_hash(file_path)
        stored_hash = self.get_hash(file_path)
        if stored_hash is None or stored_hash != current_hash:
            self.update_hash(file_path, current_hash)
            return True
        return False
    
    def remove(self, file_path: Path):
        """Remove a file from manifest."""
        key = str(file_path.absolute())
        if key in self.manifest:
            del self.manifest[key]
            self.save()


@dataclass
class Document:
    document_id: str
    source_file: str
    page_number: int
    text: str
    metadata: Dict


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_file: str
    page_number: int
    chunk_text: str
    metadata: Dict


class DocumentProcessor:
    def __init__(self, data_dir: Path = settings.data_dir, incremental: bool = True):
        self.data_dir = Path(data_dir)
        self.incremental = incremental
        self.manifest = FileHashManifest() if incremental else None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def extract_text(self, file_path: Path) -> List[Document]:
        source_file = file_path.name
        if file_path.suffix.lower() == ".pdf":
            return self._extract_pdf(file_path, source_file)
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            return self._extract_xlsx(file_path, source_file)
        return self._extract_text_file(file_path, source_file)

    def _extract_pdf(self, file_path: Path, source_file: str) -> List[Document]:
        reader = PdfReader(str(file_path))
        docs = []
        for page_index, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = normalize_text(raw)
            docs.append(Document(
                document_id=f"{file_hash(file_path)}-{page_index}",
                source_file=source_file,
                page_number=page_index,
                text=text,
                metadata={
                    "source_file": source_file,
                    "page_number": page_index,
                    "source_type": "pdf",
                    "document_hash": file_hash(file_path),
                },
            ))
        return docs

    def _extract_xlsx(self, file_path: Path, source_file: str) -> List[Document]:
        workbook = pd.read_excel(file_path, sheet_name=None)
        docs = []
        document_hash = file_hash(file_path)
        for sheet_name, frame in workbook.items():
            rows = frame.fillna("").astype(str).values.tolist()
            text = normalize_text("\n".join([" ".join(row) for row in rows]))
            docs.append(Document(
                document_id=f"{document_hash}-{sheet_name}",
                source_file=source_file,
                page_number=0,
                text=text,
                metadata={
                    "source_file": source_file,
                    "sheet_name": sheet_name,
                    "source_type": "xlsx",
                    "document_hash": document_hash,
                },
            ))
        return docs

    def _extract_text_file(self, file_path: Path, source_file: str) -> List[Document]:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        return [Document(
            document_id=f"{file_hash(file_path)}-1",
            source_file=source_file,
            page_number=1,
            text=normalize_text(raw),
            metadata={
                "source_file": source_file,
                "page_number": 1,
                "source_type": file_path.suffix.lower().lstrip('.'),
                "document_hash": file_hash(file_path),
            },
        )]

    def process_all(self, incremental: bool = True) -> tuple[List[Chunk], Dict[str, int]]:
        """Process all documents in data_dir.
        
        Returns:
            Tuple of (all_chunks, processing_stats)
            - all_chunks: List of chunks
            - processing_stats: {total_files, processed_files, skipped_files, total_chunks}
        """
        ensure_dir(self.data_dir)
        all_chunks: List[Chunk] = []
        stats = {"total_files": 0, "processed_files": 0, "skipped_files": 0, "total_chunks": 0}
        
        for file_path in sorted(self.data_dir.glob("**/*")):
            if file_path.is_dir():
                continue
            
            stats["total_files"] += 1
            
            # Check if we should process this file
            if incremental and self.manifest and not self.manifest.needs_processing(file_path):
                stats["skipped_files"] += 1
                continue
            
            stats["processed_files"] += 1
            docs = self.extract_text(file_path)
            for document in docs:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
        
        stats["total_chunks"] = len(all_chunks)
        
        # Save manifest
        if incremental and self.manifest:
            self.manifest.save()
        
        return all_chunks, stats

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunk_texts = self.splitter.split_text(document.text)
        chunks: List[Chunk] = []
        for index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_id = f"{document.document_id}-{index}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                source_file=document.source_file,
                page_number=document.page_number,
                chunk_text=normalize_text(chunk_text),
                metadata={
                    **document.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                },
            ))
        return chunks

    def save_chunks(self, chunks: List[Chunk], path: Path):
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(chunk) for chunk in chunks], handle, indent=2)

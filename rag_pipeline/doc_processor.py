import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pypdf import PdfReader

from .chunking import RecursiveCharacterTextSplitter
from .config import settings
from .utils import ensure_dir, file_hash, normalize_text, now_iso


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
    def __init__(self, data_dir: Path = settings.data_dir):
        self.data_dir = Path(data_dir)
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

    def process_all(self) -> List[Chunk]:
        ensure_dir(self.data_dir)
        all_chunks: List[Chunk] = []
        for file_path in sorted(self.data_dir.glob("**/*")):
            if file_path.is_dir():
                continue
            docs = self.extract_text(file_path)
            for document in docs:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
        return all_chunks

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

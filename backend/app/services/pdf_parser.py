from __future__ import annotations

import re
from pathlib import Path

import fitz  


class PDFParserError(RuntimeError):
    pass


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise PDFParserError(f"PDF file not found: {path}")

    try:
        doc = fitz.open(path)
        pages = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                pages.append(page_text)
        doc.close()
    except Exception as exc:
        raise PDFParserError(f"Failed to read PDF: {exc}") from exc

    if not pages:
        raise PDFParserError("No text could be extracted from the PDF.")

    return normalize_whitespace("\n\n".join(pages))
"""Concrete parser adapters."""

from ragit.parsing.parsers.docling import DoclingParser
from ragit.parsing.parsers.pymupdf4llm import PyMuPDF4LLMParser

__all__ = [
    "DoclingParser",
    "PyMuPDF4LLMParser",
]
"""
Tests for src/rag/document_upload.py.

The only LLM call in this module (enhance_description_with_llm) is
mocked, so these tests make zero real API calls.
"""

import io
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile

from src.rag.document_upload import documents


def _make_upload_file(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


class TestDocumentUploadValidation:
    def test_rejects_unsupported_file_type(self):
        bad_file = _make_upload_file("notes.docx", b"irrelevant content")

        with pytest.raises(HTTPException) as exc_info:
            documents("some description", bad_file)

        assert exc_info.value.status_code == 400
        assert "Only PDF and TXT" in exc_info.value.detail

    def test_accepts_txt_and_stores_chunks(self, fake_embeddings):
        txt_file = _make_upload_file(
            "resume.txt",
            b"This is a sample resume body used purely for testing.",
        )

        with patch(
            "src.rag.document_upload.enhance_description_with_llm",
            return_value="Enhanced description for testing.",
        ):
            result = documents("a resume", txt_file)

        assert result is True

    def test_saves_enhanced_description_to_disk(self, fake_embeddings, tmp_path):
        txt_file = _make_upload_file("resume.txt", b"Some resume content.")

        with patch(
            "src.rag.document_upload.enhance_description_with_llm",
            return_value="Enhanced description for testing.",
        ):
            documents("a resume", txt_file)

        saved = (tmp_path / "description.txt").read_text(encoding="utf-8")
        assert saved == "Enhanced description for testing."

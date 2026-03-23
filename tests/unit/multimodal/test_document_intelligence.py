from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from homie_core.multimodal.document_intelligence import DocumentIntelligence


@pytest.fixture
def txt_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(
        "Invoice\n"
        "Invoice Number: INV-001\n"
        "Date: 2025-01-15\n"
        "Bill To: Acme Corp\n"
        "Amount Due: $1,500.00\n"
        "Payment Terms: Net 30\n"
    )
    return str(p)


@pytest.fixture
def contract_txt(tmp_path):
    p = tmp_path / "contract.txt"
    p.write_text(
        "SERVICE AGREEMENT\n"
        "This Agreement is entered into by and between the parties.\n"
        "Whereas the contractor agrees to provide services under terms and conditions.\n"
        "The parties hereby agree to the following contract provisions.\n"
    )
    return str(p)


@pytest.fixture
def letter_txt(tmp_path):
    p = tmp_path / "letter.txt"
    p.write_text(
        "Dear Mr. Smith,\n"
        "Thank you for your recent inquiry. We are pleased to inform you\n"
        "that your application has been approved.\n"
        "Sincerely,\n"
        "Jane Doe\n"
    )
    return str(p)


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,age\nAlice,30\nBob,25\n")
    return str(p)


class TestDocumentIntelligence:
    def test_init_no_inference(self):
        di = DocumentIntelligence()
        assert di._infer is None

    def test_classify_invoice(self, txt_file):
        di = DocumentIntelligence()
        result = di.classify_document(txt_file)
        assert result == "invoice"

    def test_classify_contract(self, contract_txt):
        di = DocumentIntelligence()
        result = di.classify_document(contract_txt)
        assert result == "contract"

    def test_classify_letter(self, letter_txt):
        di = DocumentIntelligence()
        result = di.classify_document(letter_txt)
        assert result == "letter"

    def test_extract_key_values_heuristic(self, txt_file):
        di = DocumentIntelligence()
        kv = di.extract_key_values(txt_file)
        assert "Invoice Number" in kv
        assert kv["Invoice Number"] == "INV-001"
        assert "Bill To" in kv

    def test_extract_key_values_with_inference(self, txt_file):
        import json
        fn = MagicMock(return_value=json.dumps({"Invoice Number": "INV-001"}))
        di = DocumentIntelligence(inference_fn=fn)
        kv = di.extract_key_values(txt_file)
        assert kv["Invoice Number"] == "INV-001"

    def test_summarize_document(self, txt_file):
        di = DocumentIntelligence()
        summary = di.summarize_document(txt_file)
        assert len(summary) > 0
        assert "Invoice" in summary

    def test_summarize_with_inference(self, txt_file):
        fn = MagicMock(return_value="This is an invoice for $1,500.")
        di = DocumentIntelligence(inference_fn=fn)
        summary = di.summarize_document(txt_file)
        assert "invoice" in summary.lower()

    def test_extract_tables_unsupported_format(self, csv_file):
        di = DocumentIntelligence()
        tables = di.extract_tables(csv_file)
        assert tables == []

    def test_extract_tables_file_not_found(self):
        di = DocumentIntelligence()
        with pytest.raises(FileNotFoundError):
            di.extract_tables("/nonexistent/doc.pdf")

    def test_classify_file_not_found(self):
        di = DocumentIntelligence()
        with pytest.raises(FileNotFoundError):
            di.classify_document("/nonexistent/doc.txt")

    def test_summarize_empty_doc(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        di = DocumentIntelligence()
        summary = di.summarize_document(str(p))
        assert summary == "[Empty document]"

    def test_heuristic_key_values_skips_long_keys(self, tmp_path):
        p = tmp_path / "weird.txt"
        p.write_text("A" * 100 + ": some value\nName: Bob\n")
        di = DocumentIntelligence()
        kv = di.extract_key_values(str(p))
        assert "Name" in kv
        # The 100-char key should be skipped
        assert len([k for k in kv if len(k) > 60]) == 0

    def test_extract_tables_pdf_without_pdfplumber(self, tmp_path):
        """PDF table extraction gracefully returns empty when pdfplumber absent."""
        p = tmp_path / "fake.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        di = DocumentIntelligence()
        with patch("homie_core.multimodal.document_intelligence._HAS_PDFPLUMBER", False):
            tables = di.extract_tables(str(p))
            assert tables == []

from homie_core.screen_reader.pii_filter import PIIFilter


class TestPIIFilter:
    def setup_method(self):
        self.f = PIIFilter()

    def test_strips_email(self):
        assert "john@example.com" not in self.f.filter("Contact john@example.com for details")
        assert "[EMAIL]" in self.f.filter("Contact john@example.com for details")

    def test_strips_phone(self):
        result = self.f.filter("Call me at 555-123-4567")
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_strips_ssn(self):
        result = self.f.filter("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_strips_credit_card(self):
        result = self.f.filter("Card: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in result
        assert "[CARD]" in result

    def test_preserves_normal_text(self):
        text = "Working on the database migration in VS Code"
        assert self.f.filter(text) == text

    def test_handles_multiple_pii(self):
        text = "Email john@test.com or call 555-123-4567"
        result = self.f.filter(text)
        assert "john@test.com" not in result
        assert "555-123-4567" not in result

    def test_empty_string(self):
        assert self.f.filter("") == ""

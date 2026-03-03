from kumiho_memory.privacy import PIIRedactor


def test_pii_redaction_and_anonymize():
    redactor = PIIRedactor()
    text = "Email alice@example.com and phone 555-123-4567."
    redacted, entities = redactor.redact(text)

    assert "alice@example.com" not in redacted
    assert "[EMAIL_001]" in redacted
    assert entities["entities"]

    anonymized = redactor.anonymize_summary(text)
    assert "[email]" in anonymized.lower()

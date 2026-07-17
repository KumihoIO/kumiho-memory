import pytest

from kumiho_memory.privacy import CredentialDetectedError, PIIRedactor


def test_pii_redaction_and_anonymize():
    redactor = PIIRedactor()
    text = "Email alice@example.com and phone 555-123-4567."
    redacted, entities = redactor.redact(text)

    assert "alice@example.com" not in redacted
    assert "[EMAIL_001]" in redacted
    assert entities["entities"]

    anonymized = redactor.anonymize_summary(text)
    assert "[email]" in anonymized.lower()


# ---------------------------------------------------------------------------
# reject_credentials — modern key formats (#101)
#
# Positive fixtures are *fabricated, structurally-valid-shaped* tokens (never
# live secrets): a real key needs a >=20-char unbroken alphanumeric entropy
# run in its tail, so the strings below carry one.
# ---------------------------------------------------------------------------

# A Google API key: AIza + exactly 35 tail chars (30 digits + 5 letters).
_AIZA_KEY = "AIza" + "0123456789" * 3 + "abcde"
# A classic JWT (eyJ header . eyJ payload . base64url signature).
_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


@pytest.mark.parametrize(
    "label, text",
    [
        # Hyphenated OpenAI project/vendor keys.
        ("sk-proj",
         "OPENAI_API_KEY=sk-proj-" + "T3BlbkFJT0k" + "Abc123DEF456ghi789JKL012mnoPQRstuvWXyz01"),
        ("sk-ant",
         "sk-ant-api03-abc123DEF456ghi789JKL012mno345_pqr-stuVWX90"),
        # Classic (repaired pattern must still catch the pre-#101 shape).
        ("sk classic", "sk-abcdefghij0123456789ABCDEF"),
        # JWT.
        ("jwt", f"Authorization header carried {_JWT}"),
        # Google API key.
        ("aiza", f"GOOGLE_API_KEY={_AIZA_KEY}"),
        # Slack tokens.
        ("xoxb", "SLACK_BOT_TOKEN=xoxb-" + "123456789012-1234567890123-" + "abcdefghijklmnopqrst"),
        ("xoxp", "xoxp-1234567890-abcdefghij0123"),
        # DB connection strings with inline passwords.
        ("postgres", "DATABASE_URL=postgres://admin:s3cr3tpass@db.internal:5432/app"),
        ("postgresql", "postgresql://u:p4ssw0rd@host/db"),
        ("mysql", "mysql://root:toor@127.0.0.1:3306/mydb"),
        ("mongodb+srv", "mongodb+srv://user:secretpw@cluster0.mongodb.net"),
        ("redis pw-only", "redis://:supersecret@cache.internal:6379/0"),
        ("amqp", "amqp://guest:guestpw@rabbit:5672/"),
    ],
)
def test_reject_credentials_positive(label, text):
    with pytest.raises(CredentialDetectedError):
        PIIRedactor().reject_credentials(text)


@pytest.mark.parametrize(
    "label, text",
    [
        # Hyphenated prose must not read as an sk- key: no 20-char alnum run.
        ("sk-learn prose", "we prefer a sk-learn-based-approach-is-better model here"),
        ("hyphen prose", "this is a state-of-the-art-machine-learning-model-pipeline"),
        ("version string", "bump sk-1.2.3 up to sk-2.0.0 in the changelog"),
        ("short prefixed", "the placeholder sk-abc123 is not a real key"),
        # A JWT needs eyJ-anchored segments with real lengths.
        ("bare a.b.c", "the semantic version is a.b.c right now"),
        # DB URLs WITHOUT an inline password must stay clean.
        ("plain pg url", "connect to postgres://dbhost:5432/appdb please"),
        ("pg host-only", "postgres://host/db is the DSN"),
        ("redis fixture", 'RedisMemoryBuffer(redis_url="redis://test")'),
        ("redis host no pw", "redis://cache:6379/0 has no credentials"),
        # Google key too short.
        ("aiza short", "AIza" + "a" * 20),
        # Plain prose / project jargon.
        ("bge-m3", "defer the bge-m3 embedding migration until the release cycle"),
        ("clean prose", "The quick brown fox jumps over the lazy dog."),
    ],
)
def test_reject_credentials_negative(label, text):
    # Must not raise.
    PIIRedactor().reject_credentials(text)


def test_reject_credentials_existing_patterns_still_fire():
    """Regression guard: pre-#101 families still trip the gate."""
    r = PIIRedactor()
    for text in (
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",       # aws_access_key
        "Authorization: Bearer abcdef.ghijkl.mnopqr",   # bearer_token
        "-----BEGIN RSA PRIVATE KEY-----",              # private_key_header
        "token=ghp_" + "a" * 36,                        # github_token
        'api_key = "supersecretvalue123"',              # generic_secret_assignment
    ):
        with pytest.raises(CredentialDetectedError):
            r.reject_credentials(text)


def test_email_tld_pipe_typo_fixed():
    """privacy.py:18 no longer allows a literal '|' in the TLD class."""
    redactor = PIIRedactor()
    redacted, entities = redactor.redact("write to bob@example.com today")
    assert "bob@example.com" not in redacted
    assert entities["entities"][0]["type"] == "email"
    # A '|' is not a valid TLD char, so a pipe-bearing pseudo-address is a miss.
    _, none = redactor.redact("not an addr foo@bar.c|m here")
    assert not none["entities"]

"""Privacy utilities for PII detection and redaction.

Detection here is **probabilistic and best-effort**, not a guarantee.  The
patterns are regexes tuned for precision on common, well-known shapes; they
will miss novel, obfuscated, or split-across-lines secrets, and they may
occasionally over- or under-match.  Treat :meth:`PIIRedactor.reject_credentials`
as a defence-in-depth screen at the cloud-storage boundary, **never** as a
proof that text is credential-free.  Do not weaken upstream handling of
secrets on the assumption that this gate will catch everything.

Credential families currently covered by ``CREDENTIAL_PATTERNS``:

* AWS access key ids (``AKIA``/``ABIA``/``ACCA``/``ASIA`` + 16).
* HTTP ``Bearer`` tokens.
* OpenAI-style API keys, classic and hyphenated project/vendor forms
  (``sk-``/``pk-``/``rk-``/``ak-`` including ``sk-proj-…`` and ``sk-ant-…``).
* JSON Web Tokens (three base64url segments, ``eyJ``-anchored header+payload).
* Google API keys (``AIza`` + 35).
* Slack tokens (``xoxb``/``xoxa``/``xoxp``/``xoxr``/``xoxs`` + payload).
* PEM private-key headers.
* GitHub tokens (``ghp_``/``gho_``/``ghu_``/``ghs_``/``ghr_`` + 36).
* Database connection URLs that embed an inline password
  (``postgres``/``postgresql``/``mysql``/``mongodb(+srv)``/``redis``/``amqp``).
* Generic ``key = "value"`` secret assignments.

Notably *not* covered (non-exhaustive): passwords in prose, base64-encoded
secrets without a recognisable prefix, Azure/GCP service-account JSON, and
provider formats not listed above.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


class CredentialDetectedError(ValueError):
    """Raised when credential-like patterns are detected in text meant for cloud storage."""
    pass


class PIIRedactor:
    """Detect and redact common PII patterns."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{6}[-\s]?\d{5})\b",
        "ip_address": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    }

    CREDENTIAL_PATTERNS = {
        "aws_access_key": r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b",
        "bearer_token": r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
        # Classic ``sk-<entropy>`` AND hyphenated project/vendor forms
        # (``sk-proj-…``, ``sk-ant-…``).  Hyphens/underscores are allowed in
        # the tail so ``sk-proj-`` labels don't defeat the match, but a
        # lookahead still requires a >=20-char *unbroken alphanumeric* run
        # somewhere in the tail.  That run is the entropy signature of a real
        # key; dictionary-word prose like ``sk-learn-based-approach-is-better``
        # (longest alnum run 8) never satisfies it, so hyphenated words don't
        # false-positive.
        "api_key_generic": r"\b(?:sk|pk|rk|ak)-(?=[A-Za-z0-9_-]*[A-Za-z0-9]{20})[A-Za-z0-9_-]{20,}\b",
        # JWT: three base64url segments.  Both header and payload base64url of
        # a JSON object ``{"...`` always begin ``eyJ`` — anchoring both keeps
        # precision high (a bare ``a.b.c`` never matches).
        "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "google_api_key": r"\bAIza[0-9A-Za-z_-]{35}\b",
        "slack_token": r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b",
        "private_key_header": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b",
        # DB connection URLs — flag ONLY when an inline password is present
        # (``scheme://[user]:password@``).  The userinfo is optional so
        # password-only ``redis://:pass@host`` is caught, while password-less
        # URLs (``postgres://host/db``, ``redis://test``) are not.
        "db_url_with_password": r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]*:[^@\s]+@",
        "generic_secret_assignment": r"""(?:api[_-]?key|secret|token|password|passwd|credential)\s*[:=]\s*['"][^'"]{8,}['"]""",
    }

    def __init__(self) -> None:
        self.entity_counter: Dict[str, int] = {}

    def redact(self, text: str) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
        """Redact PII from text and return entities found."""
        redacted = text
        entities: List[Dict[str, str]] = []

        for entity_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                original = match.group(0)
                self.entity_counter[entity_type] = self.entity_counter.get(entity_type, 0) + 1
                placeholder = f"{entity_type.upper()}_{self.entity_counter[entity_type]:03d}"
                redacted = redacted.replace(original, f"[{placeholder}]")
                entities.append(
                    {
                        "type": entity_type,
                        "placeholder": placeholder,
                        "original": "[REDACTED]",
                    }
                )

        return redacted, {"entities": entities}

    def anonymize_summary(self, summary: str) -> str:
        """Anonymize summary by replacing PII with generic descriptors."""
        redacted, entities = self.redact(summary)
        for entity in entities["entities"]:
            placeholder = f"[{entity['placeholder']}]"
            descriptor = f"[{entity['type']}]"
            redacted = redacted.replace(placeholder, descriptor)
        return redacted

    def reject_credentials(self, text: str) -> None:
        """Reject text containing credential patterns.

        Should be called before storing to the cloud graph to prevent
        secrets from crossing the privacy boundary (spec section 10.4.5).

        Raises:
            CredentialDetectedError: If a credential pattern is detected.
        """
        for cred_type, pattern in self.CREDENTIAL_PATTERNS.items():
            if re.search(pattern, text):
                raise CredentialDetectedError(
                    f"Credential pattern detected ({cred_type}). "
                    "Text containing secrets must not be stored in the "
                    "cloud graph. Remove the credential and retry."
                )

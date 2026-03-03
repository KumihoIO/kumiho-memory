"""Privacy utilities for PII detection and redaction."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


class CredentialDetectedError(ValueError):
    """Raised when credential-like patterns are detected in text meant for cloud storage."""
    pass


class PIIRedactor:
    """Detect and redact common PII patterns."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{6}[-\s]?\d{5})\b",
        "ip_address": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    }

    CREDENTIAL_PATTERNS = {
        "aws_access_key": r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b",
        "bearer_token": r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
        "api_key_generic": r"\b(?:sk|pk|rk|ak)-[A-Za-z0-9]{20,}\b",
        "private_key_header": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b",
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

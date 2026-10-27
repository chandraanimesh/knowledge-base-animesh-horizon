"""
Secure Workspace Credential Encryption Service (Milestone 6G).

Provides authenticated symmetric encryption for workspace-scoped custom API credentials
using Python's `cryptography.fernet.Fernet` and `MultiFernet` for zero-downtime key rotation.

Cryptography Specification Note:
Fernet uses AES-128 in CBC mode with PKCS7 padding, authenticated with HMAC-SHA256,
and timestamp verification. It is NOT AES-256-GCM and NOT envelope encryption.
External KMS or HashiCorp Vault envelope encryption can be integrated as a future provider.

Security Invariants:
1. Encryption key is sourced exclusively from dedicated `APP_ENCRYPTION_KEY`.
2. NEVER derived from or fallen back to `JWT_SECRET` or `jwt_secret_key`.
3. If `APP_ENCRYPTION_KEY` is missing, unconfigured, or malformed, custom credential
   operations FAIL CLOSED with safe configuration errors.
4. Plaintext credentials and ciphertexts are NEVER logged or returned in public API models.
5. MultiFernet enables seamless key rotation: the first key is the primary encryption key;
   subsequent keys are decryption fallbacks for older ciphertexts.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from rag_app.core.config import get_settings

logger = logging.getLogger(__name__)


class CredentialEncryptionError(Exception):
    """Base exception for workspace credential encryption errors."""
    pass


class CredentialEncryptionConfigurationError(CredentialEncryptionError):
    """Raised when APP_ENCRYPTION_KEY is unconfigured, missing, or malformed."""
    pass


class CredentialDecryptionError(CredentialEncryptionError):
    """Raised when ciphertext cannot be decrypted (tampered, corrupted, or invalid key)."""
    pass


class CredentialEncryptionService:
    """
    Manages Fernet and MultiFernet encryption, decryption, and rotation for workspace credentials.
    """

    def __init__(self, key: Optional[str] = None) -> None:
        """
        Initializes the service with a dedicated encryption key or reads from Settings / Environment.
        Multiple comma-separated keys enable MultiFernet rotation (primary key first).
        """
        self._raw_keys: list[str] = []
        self._multi_fernet: Optional[MultiFernet] = None
        self._init_error: Optional[str] = None

        try:
            resolved_key = key
            if not resolved_key:
                settings = get_settings()
                resolved_key = getattr(settings, "app_encryption_key", None)
            if not resolved_key:
                resolved_key = os.environ.get("APP_ENCRYPTION_KEY")

            if not resolved_key or not str(resolved_key).strip():
                self._init_error = (
                    "APP_ENCRYPTION_KEY is not configured. "
                    "Custom credentials cannot be configured or decrypted."
                )
                return

            # Security Invariant 2: Explicitly verify APP_ENCRYPTION_KEY is NOT JWT_SECRET
            settings = get_settings()
            jwt_secret = getattr(settings, "jwt_secret_key", None) or os.environ.get("JWT_SECRET_KEY") or os.environ.get("JWT_SECRET")
            if jwt_secret and str(resolved_key).strip() == str(jwt_secret).strip():
                self._init_error = (
                    "CRITICAL SECURITY MISCONFIGURATION: APP_ENCRYPTION_KEY must never match JWT_SECRET. "
                    "Provide a dedicated Fernet encryption key."
                )
                logger.error("APP_ENCRYPTION_KEY matches JWT_SECRET; failing closed.")
                return

            # Parse comma-separated keys for MultiFernet rotation
            raw_tokens = [k.strip() for k in str(resolved_key).split(",") if k.strip()]
            if not raw_tokens:
                self._init_error = "APP_ENCRYPTION_KEY contains no valid tokens."
                return

            fernet_instances: list[Fernet] = []
            for token in raw_tokens:
                try:
                    # Validate key is 32 url-safe base64-encoded bytes
                    decoded = base64.urlsafe_b64decode(token.encode("utf-8"))
                    if len(decoded) != 32:
                        raise ValueError(f"Fernet key must be 32 url-safe base64-encoded bytes, got {len(decoded)}")
                    fernet_instances.append(Fernet(token.encode("utf-8")))
                    self._raw_keys.append(token)
                except Exception as ex:
                    self._init_error = f"Invalid Fernet key in APP_ENCRYPTION_KEY: {ex}"
                    logger.error("Failed to parse Fernet key: %s", ex)
                    return

            self._multi_fernet = MultiFernet(fernet_instances)
        except Exception as e:
            self._init_error = f"Unexpected error initializing CredentialEncryptionService: {e}"
            logger.error("Encryption initialization failure: %s", e)

    @property
    def is_configured(self) -> bool:
        """Returns True if at least one valid encryption key is active."""
        return self._multi_fernet is not None and self._init_error is None

    def require_configured(self) -> None:
        """Raises CredentialEncryptionConfigurationError if the service is not properly configured."""
        if not self.is_configured:
            raise CredentialEncryptionConfigurationError(
                self._init_error or "APP_ENCRYPTION_KEY is not configured or invalid."
            )

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypts a plaintext secret string using the primary Fernet key.
        Returns url-safe base64 ciphertext string.
        """
        self.require_configured()
        if not plaintext:
            raise ValueError("Cannot encrypt empty or null credential.")
        assert self._multi_fernet is not None

        try:
            cipher_bytes = self._multi_fernet.encrypt(plaintext.encode("utf-8"))
            return cipher_bytes.decode("utf-8")
        except Exception as e:
            logger.error("Encryption failure: %s", type(e).__name__)
            raise CredentialEncryptionError(f"Failed to encrypt credential: {e}") from e

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypts a ciphertext string using the primary or fallback Fernet keys.
        Returns the original plaintext secret string.
        """
        self.require_configured()
        if not ciphertext:
            raise ValueError("Cannot decrypt empty or null ciphertext.")
        assert self._multi_fernet is not None

        try:
            plain_bytes = self._multi_fernet.decrypt(ciphertext.encode("utf-8"))
            return plain_bytes.decode("utf-8")
        except InvalidToken as it:
            logger.warning("Decryption failed: InvalidToken (corrupted or tampered ciphertext)")
            raise CredentialDecryptionError(
                "Failed to decrypt credential: ciphertext is invalid, corrupted, or tampered."
            ) from it
        except Exception as e:
            logger.error("Decryption failure: %s", type(e).__name__)
            raise CredentialDecryptionError(f"Decryption error: {e}") from e

    def rotate(self, ciphertext: str) -> tuple[str, bool]:
        """
        Re-encrypts ciphertext with the current primary key if it was encrypted with an older secondary key.
        Returns (new_ciphertext, was_rotated).
        """
        self.require_configured()
        assert self._multi_fernet is not None

        try:
            rotated_bytes = self._multi_fernet.rotate(ciphertext.encode("utf-8"))
            new_ciphertext = rotated_bytes.decode("utf-8")
            was_rotated = (new_ciphertext != ciphertext)
            return new_ciphertext, was_rotated
        except InvalidToken as it:
            raise CredentialDecryptionError("Cannot rotate: token is invalid or tampered.") from it

    @classmethod
    def generate_key(cls) -> str:
        """Generates a secure 32-byte url-safe base64-encoded Fernet key."""
        return Fernet.generate_key().decode("utf-8")

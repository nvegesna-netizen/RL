"""Public-key envelope export; private keys never enter the EOS workload."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from importlib.metadata import version
from pathlib import Path
from typing import Any

from research.m4_eligible_replay.contracts import (
    canonical_json,
    digest_file,
    safe_member,
)


MAXIMUM_PAYLOAD_BYTES = 128 << 20


def crypto() -> tuple[Any, ...]:
    value = version("cryptography")
    if not re.fullmatch(r"\d+\.\d+\.\d+", value) or tuple(
        map(int, value.split("."))
    ) < (48, 0, 1):
        raise ValueError("patched cryptography >=48.0.1 required")
    # Optional runtime crypto is checked before any endpoint-producing work.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return hashes, serialization, padding, rsa, AESGCM


def generate_keypair(private_path: Path) -> bytes:
    """Create a local-only escrow key; return only the public PEM."""
    _, serialization, _, rsa, _ = crypto()
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def validate_public_key(public_pem: bytes) -> Any:
    """Reject invalid or private escrow material before any endpoint-producing work."""
    _, serialization, _, rsa, _ = crypto()
    public_key = serialization.load_pem_public_key(public_pem)
    if (
        not isinstance(public_key, rsa.RSAPublicKey)
        or public_key.key_size < 3072
        or public_key.public_numbers().e != 65537
    ):
        raise ValueError(
            "expected RSA public key of at least 3072 bits with exponent 65537"
        )
    return public_key


def seal(payload: bytes, *, public_pem: bytes, identity: dict) -> bytes:
    """Use fresh AES-256-GCM keys/nonces, with RSA-OAEP/SHA-256 key wrapping."""
    if not 1 <= len(payload) <= MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("private payload size outside envelope bound")
    hashes, serialization, padding, rsa, AESGCM = crypto()
    public_key = validate_public_key(public_pem)
    aad = canonical_json(identity)
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, payload, aad)
    wrapped = public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    return canonical_json(
        {
            "schema": "m4-private-envelope-v1",
            "identity": identity,
            "public_key_sha256": hashlib.sha256(public_pem).hexdigest(),
            "nonce": base64.b64encode(nonce).decode(),
            "wrapped_key": base64.b64encode(wrapped).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }
    )


def unseal(envelope: bytes, *, private_path: Path, expected_identity: dict) -> bytes:
    """Decrypt only after the caller's all-arm authentication gate has opened."""
    mode = private_path.lstat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError("escrow key must be an owner-only regular file")
    if len(envelope) > 2 * MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("oversized encrypted envelope")
    row = json.loads(envelope)
    if set(row) != {
        "schema",
        "identity",
        "public_key_sha256",
        "nonce",
        "wrapped_key",
        "ciphertext",
    } or (
        row["schema"] != "m4-private-envelope-v1"
        or row["identity"] != expected_identity
    ):
        raise ValueError("encrypted envelope identity differs")
    hashes, serialization, padding, rsa, AESGCM = crypto()
    key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 3072:
        raise ValueError("invalid escrow key")
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    if row["public_key_sha256"] != hashlib.sha256(public_pem).hexdigest():
        raise ValueError("wrong escrow key")
    decode = lambda value: base64.b64decode(value, validate=True)
    symmetric = key.decrypt(
        decode(row["wrapped_key"]),
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    nonce = decode(row["nonce"])
    if len(nonce) != 12 or len(symmetric) != 32:
        raise ValueError("invalid envelope key or nonce length")
    payload = AESGCM(symmetric).decrypt(
        nonce, decode(row["ciphertext"]), canonical_json(expected_identity)
    )
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("oversized private payload")
    return payload


def gated_open(
    receipts: dict[tuple[str, str], dict],
    paths: dict[tuple[str, str], Path],
    *,
    required: set[tuple[str, str]],
    expected_identities: dict[tuple[str, str], dict],
    private_path: Path,
) -> dict[tuple[str, str], bytes]:
    """Authenticate the complete ciphertext set before even loading the key."""
    if (
        set(receipts) != required
        or set(paths) != required
        or set(expected_identities) != required
    ):
        raise ValueError("not all encrypted arm outputs are authenticated")
    authenticated = {}
    for subject in sorted(required):
        receipt, path = receipts[subject], paths[subject]
        if set(receipt) != {
            "schema",
            "status",
            "identity",
            "public_key_sha256",
            "envelope_sha256",
            "plaintext_exported",
        } or (
            receipt["schema"] != "m4-encrypted-export-receipt-v1"
            or receipt["identity"] != expected_identities[subject]
            or receipt["status"] != "COMPLETE_ENCRYPTED_PRIVATE_EXPORT"
            or receipt["plaintext_exported"] is not False
        ):
            raise ValueError("private export completion identity differs")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 2 * MAXIMUM_PAYLOAD_BYTES
        ):
            raise ValueError("encrypted export path or size differs")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["envelope_sha256"]:
            raise ValueError("encrypted private export hash differs")
        envelope = json.loads(data)
        if (
            envelope["identity"] != expected_identities[subject]
            or envelope["public_key_sha256"] != receipt["public_key_sha256"]
        ):
            raise ValueError("encrypted export receipt/key binding differs")
        authenticated[subject] = data
    return {
        subject: unseal(
            authenticated[subject],
            private_path=private_path,
            expected_identity=expected_identities[subject],
        )
        for subject in sorted(required)
    }


def export_files(
    *,
    private_root: Path,
    names: list[str],
    public_pem: bytes,
    identity: dict,
    envelope_path: Path,
    receipt_path: Path,
) -> dict:
    """Export only authenticated ciphertext, never recurse into raw gradient files."""
    if envelope_path.exists() or receipt_path.exists() or len(names) != len(set(names)):
        raise FileExistsError("private export already exists or duplicates members")
    files = {}
    total = 0
    for name in names:
        relative = safe_member(name)
        path = private_root / relative
        if (
            private_root.is_symlink()
            or path.is_symlink()
            or any((private_root / p).is_symlink() for p in relative.parents)
        ):
            raise ValueError("private export forbids symlinks")
        size = path.stat().st_size
        total += size
        if total > MAXIMUM_PAYLOAD_BYTES // 2:
            raise ValueError("private export exceeds payload bound")
        files[str(relative)] = base64.b64encode(path.read_bytes()).decode()
    payload = canonical_json({"schema": "m4-private-file-bundle-v1", "files": files})
    envelope = seal(payload, public_pem=public_pem, identity=identity)
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    with envelope_path.open("xb") as stream:
        stream.write(envelope)
    receipt = {
        "schema": "m4-encrypted-export-receipt-v1",
        "status": "COMPLETE_ENCRYPTED_PRIVATE_EXPORT",
        "identity": identity,
        "public_key_sha256": hashlib.sha256(public_pem).hexdigest(),
        "envelope_sha256": digest_file(envelope_path),
        "plaintext_exported": False,
    }
    with receipt_path.open("xb") as stream:
        stream.write(canonical_json(receipt))
    return receipt

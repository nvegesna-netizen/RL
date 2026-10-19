"""Prepare hash-verified Linux crypto wheels and local-only private key escrow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tomllib
from urllib.parse import urlparse
from urllib.request import urlopen

from research.m4_eligible_replay.contracts import canonical_json
from research.m4_paired_replay.quarantine import generate_keypair


ROOT = Path(__file__).resolve().parents[3]


def prepare(output: Path) -> dict:
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    selections = (
        (
            "cryptography",
            "cryptography-48.0.1-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        ),
        (
            "cffi",
            "cffi-2.1.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        ),
        ("pycparser", None),
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    wheels = []
    for name, filename in selections:
        package = next(p for p in packages if p["name"] == name)
        candidates = [
            w
            for w in package["wheels"]
            if (filename is None and w["url"].endswith("-py3-none-any.whl"))
            or (filename is not None and w["url"].endswith("/" + filename))
        ]
        if len(candidates) != 1:
            raise ValueError("exact pinned Linux crypto wheel unavailable")
        wheel = candidates[0]
        parsed = urlparse(wheel["url"])
        if parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org":
            raise ValueError("unexpected wheel origin")
        with urlopen(wheel["url"], timeout=90) as response:
            data = response.read(wheel["size"] + 1)
        if (
            len(data) != wheel["size"]
            or "sha256:" + hashlib.sha256(data).hexdigest() != wheel["hash"]
        ):
            raise ValueError("locked Linux crypto wheel authentication failed")
        filename = Path(parsed.path).name
        with (output / filename).open("xb") as stream:
            stream.write(data)
        wheels.append(
            {
                "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "base64": base64.b64encode(data).decode(),
            }
        )
    public = generate_keypair(output / "escrow-private.pem")
    with (output / "public.pem").open("xb") as stream:
        stream.write(public)
    materials = {
        "schema": "m4-engineering-public-materials-v1",
        "public_pem_base64": base64.b64encode(public).decode(),
        "public_key_sha256": hashlib.sha256(public).hexdigest(),
        "wheels": wheels,
    }
    with (output / "public-materials.json").open("xb") as stream:
        stream.write(canonical_json(materials))
    return {
        "status": "PUBLIC_MATERIALS_READY_PRIVATE_KEY_LOCAL_ONLY",
        "public_key_sha256": materials["public_key_sha256"],
        "wheel_sha256": {w["filename"]: w["sha256"] for w in wheels},
        "private_key_uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), sort_keys=True))


if __name__ == "__main__":
    main()

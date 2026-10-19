"""Omit inert documentation and archived report data from source transport only."""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

from research.m4_eligible_replay.contracts import canonical_json, digest_file


SOURCE_HASH = "ff8cb7f09b64b6ff5d781c12c52d566c8c8821ebab559d569e322ec2fabe691f"
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".mp4",
    ".pdf",
    ".ico",
}


def omitted(member: tarfile.TarInfo) -> bool:
    if not member.isfile():
        return False
    suffix = Path(member.name).suffix.lower()
    return (
        member.name.startswith("docs/")
        and suffix in IMAGE_SUFFIXES | {".md", ".rst", ".ipynb"}
    ) or (
        member.name.startswith("reports/auto_research/")
        and suffix in IMAGE_SUFFIXES | {".md", ".json", ".tex", ".tsv", ".csv"}
    )


def inventory(path: Path) -> dict:
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name in result:
                raise ValueError("duplicate source archive member")
            stream = archive.extractfile(member) if member.isfile() else None
            result[member.name] = {
                "type": member.type.decode(),
                "mode": member.mode,
                "size": member.size,
                "mtime": member.mtime,
                "uid": member.uid,
                "gid": member.gid,
                "linkname": member.linkname,
                "sha256": hashlib.sha256(stream.read()).hexdigest() if stream else None,
            }
    return result


def validate(source: Path, projected: Path) -> dict:
    if digest_file(source) != SOURCE_HASH:
        raise ValueError("authenticated original source archive changed")
    original, actual = inventory(source), inventory(projected)
    with tarfile.open(source, "r:gz") as archive:
        removed = {m.name for m in archive.getmembers() if omitted(m)}
    expected = {name: value for name, value in original.items() if name not in removed}
    if not removed or expected != actual:
        raise ValueError(
            "projection changed retained source bytes or omitted non-documentation/report members"
        )
    return {
        "schema": "m4-inert-documentation-report-projection-v1",
        "original_source_sha256": SOURCE_HASH,
        "projected_source_sha256": digest_file(projected),
        "original_compressed_bytes": source.stat().st_size,
        "projected_compressed_bytes": projected.stat().st_size,
        "retained_member_count": len(actual),
        "retained_inventory_sha256": hashlib.sha256(canonical_json(actual)).hexdigest(),
        "all_retained_file_bytes_and_metadata_identical": True,
        "removed_member_count": len(removed),
        "removed_inventory_sha256": hashlib.sha256(
            canonical_json({name: original[name] for name in sorted(removed)})
        ).hexdigest(),
        "removed_member_examples": sorted(removed)[:10],
        "omission_rule": "documentation markdown/notebooks/images and archived auto-research report data only; never Python, YAML, TOML, or test fixtures",
        "all_python_yaml_toml_and_test_fixture_files_retained": all(
            name in actual
            for name in original
            if Path(name).suffix.lower() in {".py", ".yaml", ".yml", ".toml"}
            or name.startswith("tests/")
        ),
    }


def project(source: Path, output: Path) -> dict:
    if digest_file(source) != SOURCE_HASH:
        raise ValueError("authenticated original source archive changed")
    with tarfile.open(source, "r:gz") as original, output.open("xb") as stream:
        with gzip.GzipFile(
            filename="", fileobj=stream, mode="wb", compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
                pax_headers=original.pax_headers,
            ) as target:
                for member in original.getmembers():
                    if not omitted(member):
                        target.addfile(
                            copy.copy(member),
                            original.extractfile(member) if member.isfile() else None,
                        )
    return validate(source, output)


def replay_budget(
    *, generated_archive: Path, source: Path, projected: Path, output: Path
) -> dict:
    """Replay the exact failed generator members with only the transport replacement."""
    proof = validate(source, projected)
    old_payload = base64.b64encode(source.read_bytes())
    new_payload = base64.b64encode(projected.read_bytes())
    old_hash, new_hash = SOURCE_HASH.encode(), proof["projected_source_sha256"].encode()
    replacements = 0
    total_uncompressed = 0
    with (
        zipfile.ZipFile(generated_archive) as original,
        zipfile.ZipFile(
            output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as replay,
    ):
        for member in original.infolist():
            data = original.read(member)
            count = data.count(old_payload)
            if count:
                if count != 1 or data.count(old_hash) != 1:
                    raise ValueError("generator source transport occurrence differs")
                data = data.replace(old_payload, new_payload).replace(
                    old_hash, new_hash
                )
                replacements += count
            total_uncompressed += len(data)
            replay.writestr(member.filename, data)
    if replacements != 3:
        raise ValueError(
            "expected exactly three source-payload copies in generated archive"
        )
    cap = 50 << 20
    if output.stat().st_size >= 40 << 20 or total_uncompressed >= cap - (1 << 20):
        raise ValueError("projected generator archive lacks a robust size margin")
    return {
        "schema": "m4-generator-artifact-budget-replay-v1",
        "original_generator_archive_sha256": digest_file(generated_archive),
        "original_generator_archive_bytes": generated_archive.stat().st_size,
        "replayed_archive_sha256": digest_file(output),
        "replayed_archive_bytes": output.stat().st_size,
        "total_uncompressed_member_bytes": total_uncompressed,
        "server_archive_limit_bytes": cap,
        "all_uncompressed_members_below_limit_with_margin": True,
        "transport_replacements": replacements,
        "no_runtime_bytes_or_numerical_gates_changed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-archive", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args()
    result = {"projection": project(args.source, args.output)}
    if args.generator_archive is not None and args.replay_output is not None:
        result["artifact_budget"] = replay_budget(
            generated_archive=args.generator_archive,
            source=args.source,
            projected=args.output,
            output=args.replay_output,
        )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

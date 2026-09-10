#!/usr/bin/env python3
"""Clean-room/static validation of the non-launchable 3B preflight candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from build_preflight_candidate import AUTH_SHA, MEGATRON_SHA, PROTOCOL_SHA, SOURCE_SHA


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = (SOURCE_SHA, MEGATRON_SHA, PROTOCOL_SHA, AUTH_SHA)
    for path, digest in zip((args.source,args.megatron,args.protocol,args.authorization),expected,strict=True):
        assert sha256(path)==digest
    manifest=json.loads(args.candidate.read_bytes()); spec=manifest["spec"]
    assert spec["nodes"]==1 and spec["time_limit"]==900
    rendered=spec["script"].format(assets_dir="/tmp/m4-llama3b-preflight")
    subprocess.run(["bash","-n"],input=rendered,text=True,check=True)
    blocks=re.findall(r"<<'PY'\n(.*?)\nPY",rendered,flags=re.DOTALL)
    assert len(blocks)==5
    for index,block in enumerate(blocks): compile(block,f"<preflight-{index}>","exec")
    encoded=re.findall(r"printf %s '([A-Za-z0-9+/=]+)' \| base64 -d",rendered)
    assert tuple(hashlib.sha256(base64.b64decode(x)).hexdigest() for x in encoded)==expected
    assert "M4_LLAMA3B_LOCAL_CANDIDATE_NO_LAUNCH_AUTHORITY" in rendered
    assert "examples/run_grpo" not in rendered and "trainer.fit" not in rendered
    assert "runllm.py" not in rendered and "sbatch " not in rendered and "srun " not in rendered
    assert "AutoConfig.from_pretrained" in rendered and "AutoTokenizer.from_pretrained" in rendered
    assert "AutoModel" not in rendered
    with tempfile.TemporaryDirectory(prefix="m4-llama3b-preflight-cleanroom-") as raw:
        root=Path(raw)/"source"; root.mkdir()
        with tarfile.open(args.source,"r:gz") as archive:
            members=archive.getmembers(); names=[PurePosixPath(m.name) for m in members]
            assert len(names)==len(set(names))
            assert not any(n.is_absolute() or ".." in n.parts for n in names)
            archive.extractall(root)
        subprocess.run(
            [sys.executable,str(root/"reports/auto_research/2026-09-10-m4-llama3p2-3b-size-extension/validate_design.py"),"--output",str(Path(raw)/"validation.json")],
            cwd=root,check=True,capture_output=True,text=True,
        )
        rebuilt=Path(raw)/"rebuilt-local-candidate.json"
        subprocess.run(
            [sys.executable,str(args.builder),"--source",str(args.source),"--megatron",str(args.megatron),"--protocol",str(args.protocol),"--authorization",str(args.authorization),"--output",str(rebuilt)],
            check=True,capture_output=True,text=True,
        )
        assert rebuilt.read_bytes()==args.candidate.read_bytes()
    result={
        "schema":"m4-llama3p2-3b-no-training-preflight-local-package-validation-v1",
        "status":"PASS",
        "candidate_sha256":sha256(args.candidate),
        "source_archive_sha256":SOURCE_SHA,
        "megatron_archive_sha256":MEGATRON_SHA,
        "protocol_sha256":PROTOCOL_SHA,
        "authorization_sha256":AUTH_SHA,
        "source_members":len(members),
        "embedded_python_blocks":len(blocks),
        "deterministic_rebuild":True,
        "launch_authority_present":False,
        "launch_attempted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()

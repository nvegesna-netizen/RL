#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the Bridge-packaging-repaired downstream-quality EOS preflight v4."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

V3_MANIFEST_SHA256 = "77096eb6fe922905fda2b2560fef49212d439a7c69a79a6abf9a3665cd8222d0"
V3_AUTHORIZATION_SHA256 = (
    "8f542ee8c21a6ee11e8ad7eb37df5755f79ec7573b7caaa16db8b327db34f809"
)
V3_TERMINAL_RESULT_SHA256 = (
    "58ee78843d6967ad19d95ba59fe6532ab6f7de9d5b27c059d1883d92469f7e6f"
)
V4_AUTHORIZATION_SHA256 = (
    "76cc225ff4d09b06cfbf85f8d97c3056cda7b558a2c3e60e4a4d84f1545544a2"
)
MEGATRON_DEPENDENCY_SHA256 = (
    "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
)
BRIDGE_ARCHIVE_SHA256 = (
    "1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
)
BRIDGE_COMMIT = "573e088c9c6740082c39744e03dc5b009e730ed4"
MEGATRON_LM_COMMIT = "6513e3e23d6b5eda6a1c934990b15e804237732b"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(value: str, old: str, new: str, count: int, label: str) -> str:
    """Replace an exact number of frozen occurrences or fail closed."""
    if value.count(old) != count:
        raise RuntimeError(f"expected {count} occurrences of {label}")
    return value.replace(old, new)


def main() -> None:
    """Add only pinned Bridge packaging, authority, identity, and import gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--v3-authorization", type=Path, required=True)
    parser.add_argument("--v3-terminal-result", type=Path, required=True)
    parser.add_argument("--v4-authorization", type=Path, required=True)
    parser.add_argument("--megatron-dependency", type=Path, required=True)
    parser.add_argument("--bridge-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.v3_manifest, V3_MANIFEST_SHA256),
        (args.v3_authorization, V3_AUTHORIZATION_SHA256),
        (args.v3_terminal_result, V3_TERMINAL_RESULT_SHA256),
        (args.v4_authorization, V4_AUTHORIZATION_SHA256),
        (args.megatron_dependency, MEGATRON_DEPENDENCY_SHA256),
        (args.bridge_archive, BRIDGE_ARCHIVE_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"v4 transform input moved: {path.name}")

    authorization = json.loads(args.v4_authorization.read_bytes())
    if (
        authorization["scope"]
        != "exactly_one_bridge_packaging_repaired_credential_free_downstream_quality_no_training_eos_preflight_v4"
        or authorization["required_repair"]
        != "package_pinned_megatron_bridge_src_and_gate_actual_converter_import_before_model_access"
        or authorization["megatron_bridge_commit"] != BRIDGE_COMMIT
        or authorization["megatron_lm_commit"] != MEGATRON_LM_COMMIT
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["submission_attempt_limit"] != 1
        or authorization["eos_submission_authorized"] is not True
        or authorization["model_weight_access_authorized"] is not True
        or authorization["optimizer_steps_allowed"] != 0
    ):
        raise RuntimeError("v4 EOS preflight authority differs")
    blocked = (
        "optimizer_initialization_authorized",
        "training_authorized",
        "qualification_authorized",
        "pilot_authorized",
        "scientific_acquisition_authorized",
        "automatic_retry",
        "automatic_extension",
    )
    if any(authorization[key] for key in blocked):
        raise RuntimeError("v4 authority is not fail-closed")

    manifest = json.loads(args.v3_manifest.read_bytes())
    script = manifest["spec"]["script"]
    script = replace_exact(
        script,
        base64.b64encode(args.v3_authorization.read_bytes()).decode(),
        base64.b64encode(args.v4_authorization.read_bytes()).decode(),
        1,
        "authorization payload",
    )
    script = replace_exact(
        script,
        V3_AUTHORIZATION_SHA256,
        V4_AUTHORIZATION_SHA256,
        1,
        "authorization hash",
    )
    script = replace_exact(
        script,
        "readonly MEGATRON=/workspace/m4-downstream-quality-megatron.tar.gz\n",
        "readonly MEGATRON=/workspace/m4-downstream-quality-megatron.tar.gz\n"
        "readonly BRIDGE=/workspace/m4-downstream-quality-bridge.tar.gz\n",
        1,
        "Bridge archive declaration",
    )
    megatron_payload = base64.b64encode(args.megatron_dependency.read_bytes()).decode()
    bridge_payload = base64.b64encode(args.bridge_archive.read_bytes()).decode()
    script = replace_exact(
        script,
        f"printf %s '{megatron_payload}' | base64 -d > \"$MEGATRON\"\n",
        f"printf %s '{megatron_payload}' | base64 -d > \"$MEGATRON\"\n"
        f"printf %s '{bridge_payload}' | base64 -d > \"$BRIDGE\"\n",
        1,
        "Bridge payload insertion point",
    )
    script = replace_exact(
        script,
        f'test "$(sha256sum "$MEGATRON" | cut -d\' \' -f1)" = "{MEGATRON_DEPENDENCY_SHA256}"\n',
        f'test "$(sha256sum "$MEGATRON" | cut -d\' \' -f1)" = "{MEGATRON_DEPENDENCY_SHA256}"\n'
        f'test "$(sha256sum "$BRIDGE" | cut -d\' \' -f1)" = "{BRIDGE_ARCHIVE_SHA256}"\n',
        1,
        "Bridge hash insertion point",
    )

    authority_start = script.index('"$PYTHON" - "$AUTH" <<\'PY\'\n')
    authority_end = script.index("\nPY\nmkdir -p", authority_start) + len("\nPY")
    authority_block = """"$PYTHON" - "$AUTH" <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_bytes())
assert a["schema"]=="m4-downstream-quality-no-training-eos-preflight-v4-authorization-v1"
assert a["scope"]=="exactly_one_bridge_packaging_repaired_credential_free_downstream_quality_no_training_eos_preflight_v4"
assert a["user_authorization"]=="do the repair, gather evidence to determine how to do the repair completely. i authorize launches on eos"
assert a["predecessor_terminal_result_sha256"]=="58ee78843d6967ad19d95ba59fe6532ab6f7de9d5b27c059d1883d92469f7e6f"
assert a["predecessor_manifest_sha256"]=="77096eb6fe922905fda2b2560fef49212d439a7c69a79a6abf9a3665cd8222d0"
assert a["source_commit"]=="573dddce4cfca5fd615079c8045af05dddf80d03"
assert a["source_archive_sha256"]=="3e1cf6fb4fb287c120e7502beeedd6fb50534af3bc83b4becad610e76bb34966"
assert a["megatron_bridge_commit"]=="573e088c9c6740082c39744e03dc5b009e730ed4"
assert a["megatron_bridge_archive_sha256"]=="1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313"
assert a["megatron_lm_commit"]=="6513e3e23d6b5eda6a1c934990b15e804237732b"
assert a["megatron_dependency_sha256"]=="98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
assert a["required_repair"]=="package_pinned_megatron_bridge_src_and_gate_actual_converter_import_before_model_access"
assert a["required_launcher"]=="runllm.py --no_wait"
assert a["submission_attempt_limit"]==1
assert a["eos_submission_authorized"] is True
assert a["model_weight_access_authorized"] is True
assert a["optimizer_steps_allowed"]==0
blocked=("optimizer_initialization_authorized","training_authorized","qualification_authorized","pilot_authorized","scientific_acquisition_authorized","automatic_retry","automatic_extension")
assert not any(a[key] for key in blocked)
print("M4_DOWNSTREAM_QUALITY_EOS_PREFLIGHT_V4_AUTHORITY_PASS")
PY"""
    script = script[:authority_start] + authority_block + script[authority_end:]

    dependency_start = script.index('"$PYTHON" - "$MEGATRON" "$RUN_REPO" <<\'PY\'\n')
    dependency_end = script.index('\nPY\ncd "$RUN_REPO"', dependency_start) + len(
        "\nPY"
    )
    dependency_block = """"$PYTHON" - "$MEGATRON" "$BRIDGE" "$RUN_REPO" <<'PY'
import sys,tarfile
from pathlib import Path,PurePosixPath
megatron_archive,bridge_archive,root=map(Path,sys.argv[1:])
bridge_workspace=root/"3rdparty/Megatron-Bridge-workspace"
bridge_root=bridge_workspace/"Megatron-Bridge"
megatron_dest=bridge_root/"3rdparty"
for archive,dest,expected,prefix in (
 (bridge_archive,bridge_workspace,1060,"Megatron-Bridge/"),
 (megatron_archive,megatron_dest,726,"Megatron-LM/"),
):
 with tarfile.open(archive,"r:gz") as source:
  members=source.getmembers(); names=[PurePosixPath(m.name) for m in members]
  assert len(members)==expected
  assert len(names)==len(set(names))
  assert all(str(name)==prefix.rstrip("/") or str(name).startswith(prefix) for name in names)
  assert not any(name.is_absolute() or ".." in name.parts for name in names)
  symlinks={{name for name,member in zip(names,members,strict=True) if member.issym()}}
  assert not any(any(parent in symlinks for parent in name.parents) for name in names)
  dest.mkdir(parents=True,exist_ok=True); source.extractall(dest)
assert (bridge_root/"src/megatron/bridge/__init__.py").is_file()
assert (bridge_root/"src/megatron/bridge/models/conversion/auto_bridge.py").is_file()
assert (megatron_dest/"Megatron-LM/megatron/core/__init__.py").is_file()
print("M4_DOWNSTREAM_QUALITY_MEGATRON_DEPENDENCIES_PASS")
PY"""
    script = script[:dependency_start] + dependency_block + script[dependency_end:]
    script = replace_exact(
        script,
        'export PYTHONPATH="$RUN_REPO"\n',
        'export PYTHONPATH="$RUN_REPO/3rdparty/Megatron-Bridge-workspace/'
        "Megatron-Bridge/src:$RUN_REPO/3rdparty/Megatron-Bridge-workspace/"
        'Megatron-Bridge/3rdparty/Megatron-LM:$RUN_REPO"\n',
        1,
        "converter PYTHONPATH",
    )
    protocol_start = script.index('"$PYTHON" - "$PROTOCOL" "$PLAN" <<\'PY\'\n')
    import_gate = """"$PYTHON" - "$RUN_REPO" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
import megatron.bridge
import megatron.core
from megatron.bridge import AutoBridge
bridge_path=Path(megatron.bridge.__file__).resolve()
core_path=Path(megatron.core.__file__).resolve()
assert bridge_path.is_relative_to(root)
assert core_path.is_relative_to(root)
assert AutoBridge.__module__.startswith("megatron.bridge.")
print("M4_DOWNSTREAM_QUALITY_BRIDGE_IMPORT_PASS",bridge_path,core_path)
PY
"$PYTHON" examples/converters/convert_megatron_to_hf.py --help >/dev/null
echo M4_DOWNSTREAM_QUALITY_CONVERTER_IMPORT_PASS
"""
    script = script[:protocol_start] + import_gate + script[protocol_start:]

    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = "m4-downstream-quality-no-training-preflight-v4"
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()

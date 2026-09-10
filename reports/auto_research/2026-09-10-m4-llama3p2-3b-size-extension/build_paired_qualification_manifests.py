#!/usr/bin/env python3
"""Build the authorized paired Llama 3B neutral qualification manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import yaml


SOURCE_SHA = "1ae875a63786d687fcf46bd7fa7cb2f26b1031c3cf3c36e38114305c732820eb"
SOURCE_COMMIT = "88424bdb7bfb526e541b4a7b07830bc31028e64f"
MEGATRON_SHA = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
CONTRACT_SHA = "4e124c51a2179bd3faff5da8d34a58dc0ce33f98aa58faa2e00f1d6965944e62"
AMENDMENT_SHA = "e5ac184ea3fc1dadd1cc0cb9eb3471b991b316d2e096269e9963597074dc74b0"
RECEIPT_SHA = "6a9fc16e404d15f3f133c1fef042cfbd5cb53eb01563a4b8c794f97bd21e90c0"
PREFLIGHT_SHA = "eaa4408d27becc95d12fad7e51c89731c6106046acd0b5fe596259061d32899a"
AUTH_SHA = "90997c5ceead304efe0f535512b1c4cce49d88e457fb1fe9ded766f32cf603ff"
OLD_SOURCE_SHA = "1ec8b4cf9ca3f14f088038bfcf5851d8bd85741f0b55915227c311789b863576"
OLD_CONTRACT_SHA = "b011c846a8e876e9ea37f143d26089c7c498058a1f3e3c2461e6b317ff368fec"
OLD_PREFLIGHT_SHA = "f4fc789bb6c7cc0d61ddf4677b32f4749261f21e465d940c167cc168bce1667c"
OLD_AUTH_SHA = "4fb94ffa926ad6b12a60fb0b07b635df6920edf53e883ca71ce48277ac622f81"
CELLS = {
    "openmath": {
        "template_sha": "d2ec34ddba8725ec26971f11af5352d9f00a5351f6f0b51e934d58fea092c77e",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_openmath_neutral_qualification_v1.yaml",
        "config_sha": "15788df473dca65e96b811fc09790784f9fdb248380d152855b3b6204bc61830",
        "old_config": "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_openmath_lifecycle_derived_neutral_qualification_v4.yaml",
        "dataset": "OpenMathInstruct-2",
        "domain": "m4-llama3p2-3b-openmath-neutral-qualification-v1",
        "seed": 20261101,
        "assignment_bootstrap_seed": 20261107,
        "timing_bootstrap_seed": 20261109,
    },
    "gsm8k": {
        "template_sha": "950ffe8504fad1c8c443ad132fc504edd7dc0f00cff4d7d2dbcfa2da36e05ea6",
        "config": "examples/configs/grpo_math_3B_megatron_single_controller_m4_llama3p2_3b_gsm8k_neutral_qualification_v1.yaml",
        "config_sha": "337351dfb177f6ad61b96de4fa4f859a3bbaaaacb966e4ff4d7dad63fb469fd5",
        "old_config": "examples/configs/grpo_math_1B_megatron_single_controller_m4_llama3p2_1b_gsm8k_lifecycle_derived_neutral_qualification_v4.yaml",
        "dataset": "gsm8k",
        "domain": "m4-llama3p2-3b-gsm8k-neutral-qualification-v1",
        "seed": 20261102,
        "assignment_bootstrap_seed": 20261108,
        "timing_bootstrap_seed": 20261110,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def replace_payload(script: str, variable: str, path: Path) -> str:
    pattern = rf"printf %s '[A-Za-z0-9+/=]+' \| base64 -d > \"\${variable}\""
    replacement = f"printf %s '{b64(path)}' | base64 -d > \"${variable}\""
    result, count = re.subn(pattern, replacement, script)
    if count != 1:
        raise RuntimeError(f"expected one {variable} payload, found {count}")
    return result


def replace_block(script: str, start: str, end: str, replacement: str) -> str:
    left = script.index(start)
    right = script.index(end, left)
    return script[:left] + replacement + script[right:]


def escaped(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def build(cell: str, args: argparse.Namespace) -> dict[str, object]:
    value = CELLS[cell]
    manifest = yaml.safe_load(args.template.read_bytes())
    script = manifest["spec"]["script"]
    for old, new in (
        ("m4-llama-v4-qualification-repo", "m4-llama3b-qualification-repo"),
        ("m4-llama-v3-scientific-source.tar.gz", "m4-llama3b-scientific-source.tar.gz"),
        ("m4-llama-v4-qualification-contract.json", "m4-llama3b-qualification-contract.json"),
        ("m4-llama-v3-bootstrap-terminal-result.json", "m4-llama3b-preflight-terminal-result.json"),
        ("m4-llama-v4-qualification-authorization.json", "m4-llama3b-qualification-authorization.json"),
        ("M4_LLAMA_V4_", "M4_LLAMA3B_"),
        (OLD_SOURCE_SHA, SOURCE_SHA),
        (OLD_CONTRACT_SHA, CONTRACT_SHA),
        (OLD_PREFLIGHT_SHA, PREFLIGHT_SHA),
        (OLD_AUTH_SHA, AUTH_SHA),
        ("5804fe81d706f8ad9a3db4d2f41782a742710c51", SOURCE_COMMIT),
        (value["old_config"], value["config"]),
        ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ):
        script = script.replace(old, new)
    for variable, path in (
        ("SOURCE", args.source),
        ("CONTRACT", args.contract),
        ("PREFLIGHT", args.preflight),
        ("AUTH", args.authorization),
    ):
        script = replace_payload(script, variable, path)

    evidence_start = '"$PYTHON" - "$CONTRACT" "$RECEIPT" "$PREFLIGHT" "$AUTH" <<\'PY\'\n'
    config_start = '"$PYTHON" - <<\'PY\'\nfrom omegaconf import OmegaConf\n'
    evidence = escaped(f'''"$PYTHON" - "$CONTRACT" "$RECEIPT" "$PREFLIGHT" "$AUTH" <<'PY'
import hashlib,json,sys
from pathlib import Path
contract,receipt,preflight,auth=[json.loads(Path(path).read_bytes()) for path in sys.argv[1:]]
assert contract["schema"]=="m4-llama3p2-3b-paired-neutral-qualification-execution-contract-v1"
assert contract["status"]=="FROZEN_BEFORE_QUALIFICATION_DATA"
assert contract["prospective_protocol_sha256"]=="4f2a5e496030e836b3449697928f4f363d7d36e3cec668105b2cdcc701b9814c"
assert contract["trainer_steps_each"]==64 and contract["evaluable_start_versions"]==[8,55]
assert contract["terminal_guard_start_versions"]==[56,64]
assert contract["minimum_assignment_projection_lower_95"]==5000.0
assert contract["maximum_projected_runtime_upper_95_seconds"]==12600.0
assert contract["scheduler_cap_seconds_each"]==14400
assert contract["qualification_data_allowed_in_causal_estimator"] is False
assert contract["acquisition_authorized"] is False
assert contract["automatic_retry"] is False and contract["automatic_extension"] is False
assert receipt["megatron_lm"]["archive_sha256"]=="{MEGATRON_SHA}"
assert preflight["schema"]=="m4-llama3p2-3b-no-training-preflight-terminal-result-v1"
assert preflight["status"]=="GREEN" and preflight["result"]["training_started"] is False
assert preflight["result"]["model"]=="meta-llama/Llama-3.2-3B-Instruct"
assert auth=={json.dumps(json.loads(args.authorization.read_bytes()), sort_keys=True, separators=(',', ':'))}
registered=contract["cells"]["{cell}"]
assert registered["config"]=="{value['config']}"
assert registered["config_sha256"]=="{value['config_sha']}"
assert registered["assignment_domain"]=="{value['domain']}"
assert registered["assignment_seed"]=={value['seed']}
assert registered["assignment_bootstrap_seed"]=={value['assignment_bootstrap_seed']}
assert registered["timing_bootstrap_seed"]=={value['timing_bootstrap_seed']}
assert hashlib.sha256(Path(registered["config"]).read_bytes()).hexdigest()==registered["config_sha256"]
print("M4_LLAMA3B_{cell.upper()}_QUALIFICATION_EVIDENCE_PASS")
PY
''')
    script = replace_block(script, evidence_start, config_start, evidence)

    run_start = "readonly START_NS="
    config = escaped(f'''"$PYTHON" - <<'PY'
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,register_omegaconf_resolvers
register_omegaconf_resolvers()
c=OmegaConf.to_container(load_config("{value['config']}"),resolve=True)
assert c["grpo"]["max_num_steps"]==64
assert c["policy"]["model_name"]=="meta-llama/Llama-3.2-3B-Instruct"
assert c["policy"]["tokenizer"]["name"]=="meta-llama/Llama-3.2-3B-Instruct"
assert c["cluster"]["num_nodes"]==1 and c["cluster"]["gpus_per_node"]==2
r=c["async_rl"]["controlled_release_delay"]
assert r["enabled"] is True and r["seed"]=={value['seed']} and r["assignment_domain"]=="{value['domain']}"
assert r["arms"]==[{{"label":"neutral","delay_seconds":0.0,"mass":1}}]
assert c["async_rl"]["gradient_opportunity_audit"]["enabled"] is False
assert c["async_rl"]["lifecycle_derived_opportunity_audit"]["enabled"] is True
assert c["data"]["train"]["dataset_name"]=="{value['dataset']}"
validate_single_controller_config(MasterConfig(**c))
print("M4_LLAMA3B_{cell.upper()}_QUALIFICATION_CONFIG_PASS")
PY
''')
    script = replace_block(script, config_start, run_start, config)

    analysis_start = '"$PYTHON" - "$LIFECYCLE" "$OPPORTUNITY" "$DUTY" "$DERIVATION" "$RESULT" "$START_NS" "$END_NS" <<\'PY\'\n'
    hashes_start = '(cd "$ASSETS"; sha256sum '
    analysis = escaped(f'''"$PYTHON" - "$LIFECYCLE" "$OPPORTUNITY" "$DUTY" "$DERIVATION" "$RESULT" "$START_NS" "$END_NS" <<'PY'
import collections,json,pathlib,sys
from tools.m4_llama3b_qualification import assess_llama3b_qualification
from tools.neutral_qualification_lifecycle import assess_neutral_qualification_lifecycle
from tools.opportunity_ledger_join import LedgerJoinProtocol,ReleaseArm,join_opportunity_ledgers
lp,op,dp,xp,rp=map(pathlib.Path,sys.argv[1:6]); start,end=map(int,sys.argv[6:])
def rows(path):
 raw=path.read_bytes(); assert raw.endswith(b"\\n")
 return [json.loads(line) for line in raw.splitlines()]
lifecycle=rows(lp); opportunity=rows(op); duty=json.loads(dp.read_bytes()); derivation=json.loads(xp.read_bytes())
groups=[row for row in opportunity if row.get("event_type")=="group"]
steps=[row for row in opportunity if row.get("event_type")=="train_step_completed"]
group_ids={{row["group_id"] for row in groups}}; rewards={{s["reward"] for row in groups for s in row["siblings"]}}
assessment=assess_neutral_qualification_lifecycle(lifecycle,opportunity_group_ids=group_ids)
protocol=LedgerJoinProtocol(assignment_domain="{value['domain']}",assignment_seed={value['seed']},arms=(ReleaseArm("neutral",0.0,1),),primary_start_version=8,primary_end_version=55,siblings_per_group=8,train_batch_size=32)
joined=join_opportunity_ledgers(protocol=protocol,lifecycle_rows=lifecycle,opportunity_rows=opportunity)
versions=collections.Counter(row["start_weight_version"] for row in joined)
advances=sorted((row for row in lifecycle if row.get("stage")=="learner_version_advanced"),key=lambda row:row["learner_weight_version"])
advance_time={{row["learner_weight_version"]:row["timestamp_ns"] for row in advances}}
intervals=[(advance_time[version+1]-advance_time[version])/1e9 for version in range(8,56)]
duration=(end-start)/1e9
qualification=assess_llama3b_qualification(versions,intervals,total_qualification_runtime_seconds=duration,assignment_bootstrap_seed={value['assignment_bootstrap_seed']},timing_bootstrap_seed={value['timing_bootstrap_seed']}).to_dict()
expected_corrected=max(0,duty["raw_observer_ns"]-duty["observation_count"]*duty["paired_clock_overhead_ns"])
checks={{
 "complete_steps":len(steps)==64 and [row["learner_version"] for row in steps]==list(range(1,65)) and [row["learner_weight_version"] for row in advances]==list(range(1,65)),
 "neutral_lifecycle":assessment["supported"],
 "group_ids_unique":len(groups)==len(group_ids),
 "binary_reward_support":rewards=={{0.0,1.0}},
 "exact_evaluable_window":set(versions)==set(range(8,56)),
 "strict_join":len(joined)==sum(8<=row["start_weight_version"]<=55 for row in groups),
 "assignment_support":qualification["assignment_support_passed"],
 "timing_support":qualification["timing_support_passed"],
 "joint_qualification_gate":qualification["qualified"],
 "lifecycle_duty":duty["corrected_observer_ns"]==expected_corrected and duty["corrected_observer_duty"]<=0.01,
 "derivation":derivation["completed_step_count"]==64 and derivation["group_count"]==len(groups) and derivation["derivation_method"]=="lifecycle_derived_grpo_opportunity_v1",
}}
value={{"schema":"m4-llama3p2-3b-neutral-qualification-result-v1","cell":"{cell}","source_commit":"{SOURCE_COMMIT}","trainer_steps":64,"gpus":2,"wall_time_seconds":duration,"opportunity_group_count":len(groups),"strict_join_assignment_count":len(joined),"lifecycle_event_count":len(lifecycle),"corrected_lifecycle_duty":duty["corrected_observer_duty"],"reward_values":sorted(rewards),"qualification":qualification,"checks":checks,"qualification_complete":all(checks.values()),"causal_estimate_produced":False,"acquisition_started":False,"automatic_retry":False,"automatic_extension":False}}
rp.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
assert all(checks.values()),checks
print("M4_LLAMA3B_{cell.upper()}_NEUTRAL_QUALIFICATION_GREEN")
PY
''')
    script = replace_block(script, analysis_start, hashes_start, analysis)
    manifest["spec"]["script"] = script
    manifest["spec"]["name"] = f"m4-llama3b-{cell}-neutral-qualification-v1"
    manifest["spec"]["time_limit"] = 14400
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = CELLS[args.cell]
    for path, expected in (
        (args.template, value["template_sha"]),
        (args.source, SOURCE_SHA),
        (args.megatron, MEGATRON_SHA),
        (args.contract, CONTRACT_SHA),
        (args.amendment, AMENDMENT_SHA),
        (args.receipt, RECEIPT_SHA),
        (args.preflight, PREFLIGHT_SHA),
        (args.authorization, AUTH_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen qualification input moved: {path.name}")
    manifest = build(args.cell, args)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False, width=10**9))
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()

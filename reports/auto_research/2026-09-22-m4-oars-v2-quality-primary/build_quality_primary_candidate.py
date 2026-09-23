#!/usr/bin/env python3
"""Build one self-contained OARS-v2 quality-primary acquisition manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

SOURCE_COMMIT = "50443383825adeddca9cd1417c9dd261d8413670"
SOURCE_SHA256 = "3bab95936ff2e419532f3816ea96a53c3917b4716bccd460885d2606725d5394"
MEGATRON_SHA256 = "98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2"
PROTOCOL_SHA256 = "e0e0b26cc1a592bf14c4b0f77d990662f6c3ff3e09bfb6a772a3efbf149b9d2d"
RUN_MANIFEST_SHA256 = "44fe686d444545965afd6964387ae662c2a7f640512b1262da19b90ddfabd20f"
ANALYSIS_PLAN_SHA256 = "af09cbcc891ebf1f6eb1287defd04e2708aec8ec8c8d0ba7592975c7564e0500"
QUALIFICATION_GATE_SHA256 = (
    "7d2660bc1d1e651f83b40e037ea1890c5a3389681536f20fc8f5e389e4afc470"
)
AUTHORIZATION_SHA256 = (
    "baa75d118ea188d293f9d22b85f37f79aa76750f899cd3ce87378a7df805bb07"
)
TEMPLATE_SHA256 = "d52ee6c2474c3989ca61cca209e3e9a3f82273277e28e1f96bf0c8d1dd2c9ee4"
CONFIG_PATH = (
    "examples/configs/"
    "grpo_math_1B_megatron_single_controller_m4_oars_v2_controlled_frontier.yaml"
)
CONFIG_SHA256 = "fac9bc021c504029aa378d855132afe00635432c575def744302e5276ccb7c8e"
IMAGE_PATH = (
    "/lustre/fsw/coreai_dlalgo_ci/nvegesna/nemo_rl_images/"
    "nemo-rl-nightly-5802754.sqsh"
)
IMAGE_COMMIT = "ae07eafe8035b5b2e84efa7234e70e7fd7e493c1"
TRANSFERQUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def rendered_fragment(value: str) -> str:
    """Escape braces for JET's assets_dir format pass."""
    return (
        value.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{assets_dir}}", "{assets_dir}")
    )


def replace_block(script: str, index: int, value: str) -> str:
    matches = list(re.finditer(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL))
    if len(matches) != 7:
        raise RuntimeError("template embedded Python topology differs")
    match = matches[index]
    return script[: match.start(1)] + rendered_fragment(value) + script[match.end(1) :]


def validate_authorization(path: Path) -> None:
    authorization = json.loads(path.read_bytes())
    required_true = (
        "credential_free",
        "eos_submission_authorized",
        "model_weight_access_authorized",
        "optimizer_initialization_authorized",
        "training_authorized",
        "scientific_acquisition_authorized",
        "complete_outcome_embargo_until_all_54_authenticated",
    )
    if (
        authorization["schema"]
        != "m4-oars-v2-quality-primary-acquisition-authorization-v1"
        or authorization["scope"]
        != "execute_exactly_the_frozen_54_run_quality_primary_acquisition"
        or authorization["runtime_source_commit"] != SOURCE_COMMIT
        or authorization["protocol_sha256"] != PROTOCOL_SHA256
        or authorization["run_manifest_sha256"] != RUN_MANIFEST_SHA256
        or authorization["analysis_plan_sha256"] != ANALYSIS_PLAN_SHA256
        or authorization["qualification_gate_sha256"] != QUALIFICATION_GATE_SHA256
        or authorization["run_count"] != 54
        or authorization["required_launcher"] != "runllm.py --no_wait"
        or authorization["sequence_policy"]
        != "strict_global_sequence_one_run_at_a_time"
        or authorization["submission_attempt_limit_per_identity"] != 1
        or authorization["total_submission_attempt_limit"] != 54
        or authorization["gpus_per_run"] != 2
        or authorization["scheduler_time_limit_seconds"] != 14400
        or authorization["queue_deadline_override"] is not None
        or not all(authorization[key] for key in required_true)
        or authorization["automatic_retry"]
        or authorization["automatic_replacement"]
        or authorization["automatic_extension"]
    ):
        raise RuntimeError("quality-primary acquisition authority differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--qualification-gate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = (
        (args.template, TEMPLATE_SHA256),
        (args.source, SOURCE_SHA256),
        (args.megatron, MEGATRON_SHA256),
        (args.protocol, PROTOCOL_SHA256),
        (args.run_manifest, RUN_MANIFEST_SHA256),
        (args.analysis_plan, ANALYSIS_PLAN_SHA256),
        (args.qualification_gate, QUALIFICATION_GATE_SHA256),
        (args.authorization, AUTHORIZATION_SHA256),
    )
    for path, digest in expected:
        if sha256(path) != digest:
            raise RuntimeError(f"frozen package input moved: {path.name}")
    validate_authorization(args.authorization)

    run_manifest = json.loads(args.run_manifest.read_bytes())
    runs = run_manifest["runs"]
    if (
        run_manifest["status"] != "SEALED_NOT_AUTHORIZED_FOR_SUBMISSION"
        or run_manifest["run_count"] != 54
        or [run["global_sequence"] for run in runs] != list(range(1, 55))
    ):
        raise RuntimeError("sealed run manifest differs")
    for index, item in enumerate(runs):
        expected_predecessor = None if index == 0 else runs[index - 1]["identity"]
        if item["predecessor"] != expected_predecessor:
            raise RuntimeError("predecessor chain differs")
    run = next((item for item in runs if item["identity"] == args.identity), None)
    if run is None:
        raise RuntimeError("identity is not registered")

    template = json.loads(args.template.read_bytes())
    if (
        template["spec"]["name"]
        != "m4-oars-confirmatory-p01-fifo-acquisition"
        or template["spec"]["time_limit"] != 14400
        or template["spec"]["image_source"] != {"local_path": IMAGE_PATH}
        or "sbatch_additional_flags" in template["spec"]
    ):
        raise RuntimeError("validated template boundary differs")
    candidate = json.loads(args.template.read_bytes())
    stem = f"m4-oars-v2-quality-primary-{run['identity']}"
    script = candidate["spec"]["script"].replace(
        "m4-oars-confirmatory-p01-fifo", stem
    )

    paths = (
        args.source,
        args.megatron,
        args.protocol,
        args.run_manifest,
        args.analysis_plan,
        args.qualification_gate,
        args.authorization,
    )
    encoded = list(
        re.finditer(
            r"(printf %s ')[A-Za-z0-9+/=]+(' \| base64 -d > \"\$[A-Z_]+\")",
            script,
        )
    )
    if len(encoded) != 7:
        raise RuntimeError("template payload topology differs")
    for match, path in reversed(list(zip(encoded, paths, strict=True))):
        script = (
            script[: match.start()]
            + match.group(1)
            + payload(path)
            + match.group(2)
            + script[match.end() :]
        )

    old_hashes = (
        "1d9e17a430fc1a184d767413de829ad9ee1375091485b5e7e0b24572f61f7690",
        MEGATRON_SHA256,
        "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd",
        "1f59d8cf3339d70b58b084af48bd2f192a880d7a43fca8190fb1d54649db5b39",
        "66431cf76f3314b1bc1a0d6ece2735f9842e4790ed3eb673322c58b84d268d58",
        "d359c5536147b74c4255a3edf090d98460cff3c8fae4507ac9dcdef15ec43ca9",
        "90e48c344e5056121a020e16e0b49db0203026a8e524c73bf8101069b154384a",
    )
    new_hashes = (
        SOURCE_SHA256,
        MEGATRON_SHA256,
        PROTOCOL_SHA256,
        RUN_MANIFEST_SHA256,
        ANALYSIS_PLAN_SHA256,
        QUALIFICATION_GATE_SHA256,
        AUTHORIZATION_SHA256,
    )
    for old, new in zip(old_hashes, new_hashes, strict=True):
        script = script.replace(old, new)

    authority = f'''import hashlib,json,sys
from pathlib import Path
p_path,r_path,n_path,q_path,a_path=map(Path,sys.argv[1:])
p=json.loads(p_path.read_bytes()); r=json.loads(r_path.read_bytes()); n=json.loads(n_path.read_bytes()); q=json.loads(q_path.read_bytes()); a=json.loads(a_path.read_bytes())
assert hashlib.sha256(p_path.read_bytes()).hexdigest()=="{PROTOCOL_SHA256}"
assert hashlib.sha256(r_path.read_bytes()).hexdigest()=="{RUN_MANIFEST_SHA256}"
assert hashlib.sha256(n_path.read_bytes()).hexdigest()=="{ANALYSIS_PLAN_SHA256}"
assert hashlib.sha256(q_path.read_bytes()).hexdigest()=="{QUALIFICATION_GATE_SHA256}"
assert hashlib.sha256(a_path.read_bytes()).hexdigest()=="{AUTHORIZATION_SHA256}"
assert p["schema"]=="m4-oars-v2-quality-primary-protocol-v1" and p["design"]["total_run_count"]==54
assert r["run_count"]==54 and q["status"]=="PASS_ACQUISITION_GATE_READY_NOT_LAUNCHED"
assert n["primary"]["interval"]["df"]==17 and n["primary"]["robustness"]["enumeration_count"]==262144
assert a["schema"]=="m4-oars-v2-quality-primary-acquisition-authorization-v1"
required=("credential_free","eos_submission_authorized","model_weight_access_authorized","optimizer_initialization_authorized","training_authorized","scientific_acquisition_authorized","complete_outcome_embargo_until_all_54_authenticated")
assert all(a[key] for key in required)
forbidden=("automatic_retry","automatic_replacement","automatic_extension")
assert not any(a[key] for key in forbidden)
assert a["required_launcher"]=="runllm.py --no_wait" and a["sequence_policy"]=="strict_global_sequence_one_run_at_a_time"
assert a["submission_attempt_limit_per_identity"]==1 and a["total_submission_attempt_limit"]==54
run=next(item for item in r["runs"] if item["identity"]=="{run['identity']}")
assert run["global_sequence"]=={run['global_sequence']} and run["block"]=={run['block']}
assert run["arm"]=="{run['arm']}" and run["mode"]=="{run['mode']}"
assert run["actuation_scorer"]=={run['actuation_scorer']!r}
assert run["training_seed"]=={run['training_seed']} and run["assignment_seed"]=={run['assignment_seed']}
assert run["assignment_domain"]=="{run['assignment_domain']}"
print("M4_OARS_V2_QUALITY_PRIMARY_AUTHORITY_PASS")
'''
    script = replace_block(script, 0, authority)

    overrides = [
        f"grpo.seed={run['training_seed']}",
        f"async_rl.controlled_release_delay.seed={run['assignment_seed']}",
        (
            "async_rl.controlled_release_delay.assignment_domain="
            f"{run['assignment_domain']}"
        ),
        f"async_rl.opportunity_at_risk_v2_shadow.mode={run['mode']}",
        (
            "async_rl.opportunity_at_risk_v2_shadow.actuation_scorer="
            + ("null" if run["actuation_scorer"] is None else run["actuation_scorer"])
        ),
        "data_plane.actor_runtime_env_mode=inherit_baked_single_node",
        "async_rl.terminal_policy_export.enabled=true",
        "async_rl.terminal_policy_export.output_dir={assets_dir}/terminal-policy",
    ]
    config = f'''import json,sys,tomllib
from importlib.metadata import distribution
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig,validate_single_controller_config
from nemo_rl.utils.config import load_config,parse_hydra_overrides,register_omegaconf_resolvers
path=Path(sys.argv[1]); overrides={json.dumps(overrides)}
register_omegaconf_resolvers(); raw=parse_hydra_overrides(load_config(path),overrides); config=MasterConfig(**OmegaConf.to_container(raw,resolve=True))
assert config.grpo.max_num_steps==64 and config.grpo.max_num_epochs==2 and config.grpo.seed=={run['training_seed']}
assert config.policy["model_name"]=="meta-llama/Llama-3.2-1B-Instruct"
assert config.data["train"]["dataset_name"]=="gsm8k" and config.data["train"]["split"]=="train"
assert config.cluster["num_nodes"]==1 and config.cluster["gpus_per_node"]==2
assert config.async_rl.sampler.name=="weight_fifo" and config.async_rl.sampler.selection_candidate_watermark==8
assert config.async_rl.sampler.max_staleness_versions==1
v2=config.async_rl.opportunity_at_risk_v2_shadow
assert v2.enabled and v2.mode=="{run['mode']}" and v2.actuation_scorer=={run['actuation_scorer']!r}
assert v2.candidate_window_policy=="controlled_frontier" and v2.minimum_service_multiplier==0.98 and v2.maximum_service_multiplier==1.02
assert config.async_rl.controlled_release_delay.seed=={run['assignment_seed']}
assert config.async_rl.controlled_release_delay.assignment_domain=="{run['assignment_domain']}"
assert config.async_rl.terminal_policy_export.enabled and not config.checkpointing["enabled"]
assert config.data_plane["actor_runtime_env_mode"]=="inherit_baked_single_node"
locked=tomllib.loads(Path("uv.lock").read_text()); tq=[item for item in locked["package"] if item["name"].lower()=="transferqueue"]
assert len(tq)==1 and tq[0]["source"]["git"].endswith("#{TRANSFERQUEUE_COMMIT}")
direct=json.loads(distribution("TransferQueue").read_text("direct_url.json")); assert direct["vcs_info"]["commit_id"]=="{TRANSFERQUEUE_COMMIT}"
validate_single_controller_config(config)
print("M4_OARS_V2_QUALITY_PRIMARY_{run['identity'].replace('-', '_').upper()}_CONFIG_PASS")
'''
    script = replace_block(script, 3, config)
    old_config = (
        'readonly CONFIG="examples/configs/'
        'grpo_math_1B_megatron_single_controller_m4_oars_qualification_fifo.yaml"\n'
        'test "$(sha256sum "$CONFIG" | cut -d\' \' -f1)" = '
        '"17ad965489aa9595d266295125880fe55edb4af303bd95aed66a61034c943c9b"'
    )
    new_config = (
        f'readonly CONFIG="{CONFIG_PATH}"\n'
        f'test "$(sha256sum "$CONFIG" | cut -d\' \' -f1)" = "{CONFIG_SHA256}"'
    )
    if script.count(old_config) != 1:
        raise RuntimeError("template config boundary differs")
    script = script.replace(old_config, new_config)

    scorer = run["actuation_scorer"]
    analyzer = (
        'tools/m4_oars_v2_controlled_frontier_qualification.py '
        if scorer is None
        else f'tools/m4_oars_v2_actuation_qualification.py --scorer "{scorer}" '
    )
    training = f'''readonly START_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
set +e
"$PYTHON" examples/run_grpo_single_controller.py --config "$CONFIG" \
  "grpo.seed={run['training_seed']}" \
  "async_rl.controlled_release_delay.seed={run['assignment_seed']}" \
  "async_rl.controlled_release_delay.assignment_domain={run['assignment_domain']}" \
  "async_rl.opportunity_at_risk_v2_shadow.mode={run['mode']}" \
  "async_rl.opportunity_at_risk_v2_shadow.actuation_scorer={'null' if scorer is None else scorer}" \
  "data_plane.actor_runtime_env_mode=inherit_baked_single_node" \
  "async_rl.lifecycle_audit_path=$LIFECYCLE" \
  "async_rl.gradient_opportunity_audit.output_path=$OPPORTUNITY" \
  "async_rl.gradient_opportunity_audit.observer_duty_path=$DUTY" \
  "async_rl.opportunity_at_risk_v2_shadow.output_path=$OARS" \
  "async_rl.terminal_policy_export.enabled=true" \
  "async_rl.terminal_policy_export.output_dir=$EXPORT" \
  "logger.log_dir=$METRICS" 2>&1 | tee "$RUN_LOG"
readonly TRAIN_RC=${{PIPESTATUS[0]}}
set -e
readonly END_NS=$("$PYTHON" -c 'import time; print(time.time_ns())')
if test "$TRAIN_RC" -ne 0; then
  printf '{{"identity":"{run['identity']}","schema":"m4-oars-v2-quality-primary-arm-result-v1","scientific_outcome_acquisition":true,"status":"TRAINING_FAILURE","training_exit_code":%s,"training_quality_analyzed":false}}\\n' "$TRAIN_RC" > "$RESULT"
  exit "$TRAIN_RC"
fi
grep -Fq "SC run complete: {{'train_steps': 64, 'trainer_version': 64" "$RUN_LOG"
"$PYTHON" {analyzer}--oars "$OARS" --lifecycle "$LIFECYCLE" --observer-duty "$DUTY" --source-commit "{SOURCE_COMMIT}" --run-start-ns "$START_NS" --run-end-ns "$END_NS" --output "$SYSTEMS"
'''
    start = script.index("readonly START_NS=")
    end = script.index("readonly ITER=", start)
    script = script[:start] + rendered_fragment(training) + script[end:]

    expected_status = (
        "PASS_CONTROLLED_FRONTIER_SYSTEMS_READY"
        if scorer is None
        else "PASS_OARS_V2_ACTUATION_QUALIFIED"
    )
    outcome = f'''import hashlib,json,math,sys
from pathlib import Path
systems_path,oars_path,evaluation_path,prompts_path,result_path=map(Path,sys.argv[1:])
systems=json.loads(systems_path.read_bytes()); rows=[json.loads(line) for line in oars_path.read_text().splitlines()]; header,*decisions=rows
assert systems["status"]=="{expected_status}" and len(decisions)==64 and all(systems["checks"].values())
assert header["mode"]=="{run['mode']}" and all(row["mode"]=="{run['mode']}" for row in decisions)
scorer={scorer!r}
if scorer is None:
 selected=[{{"l1":row["baseline_l1"],"tokens":row["baseline_valid_actor_tokens"],"overlap":4}} for row in decisions]
else:
 assert all(row["proposal_matches_actual"] is True and row["actuation_scorer"]==scorer for row in decisions)
 selected=[{{"l1":row["proposals"][scorer]["proposed_l1"],"tokens":row["proposals"][scorer]["proposed_valid_actor_tokens"],"overlap":row["proposals"][scorer]["fifo_overlap_count"]}} for row in decisions]
retained_l1=math.fsum(float(row["l1"]) for row in selected)/(64*4)
tokens=math.fsum(float(row["tokens"]) for row in selected)/64
mean_fifo_overlap=math.fsum(float(row["overlap"]) for row in selected)/64
evaluation=json.loads(evaluation_path.read_bytes())["evaluation_data"]; prompt_manifest=json.loads(prompts_path.read_bytes()); assert len(evaluation)==len(prompt_manifest["row_sha256"])==1319
rewards=[float(row["reward"]) for row in evaluation]; assert all(reward in (0.0,1.0) for reward in rewards)
value={{"schema":"m4-oars-v2-quality-primary-arm-result-v1","status":"PASS","protocol_sha256":"{PROTOCOL_SHA256}","run_manifest_sha256":"{RUN_MANIFEST_SHA256}","analysis_plan_sha256":"{ANALYSIS_PLAN_SHA256}","source_commit":"{SOURCE_COMMIT}","identity":"{run['identity']}","global_sequence":{run['global_sequence']},"block":{run['block']},"arm":"{run['arm']}","mode":"{run['mode']}","actuation_scorer":scorer,"training_seed":{run['training_seed']},"assignment_seed":{run['assignment_seed']},"assignment_domain":"{run['assignment_domain']}","scientific_outcome_acquisition":True,"training_quality_analyzed":True,"systems":{{"qualification_status":systems["status"],"checks":systems["checks"],"time_to_update_64_seconds":systems["runtime_seconds"],"gradient_observer_duty":systems.get("gradient_observer_duty"),"scheduler_decision_duty":systems.get("oars_v2_decision_duty",systems.get("oars_observer_duty"))}},"mechanism":{{"retained_l1_per_selected_group":retained_l1,"valid_actor_tokens_per_update":tokens,"mean_fifo_overlap":mean_fifo_overlap}},"outcomes":{{"terminal_gsm8k_accuracy":math.fsum(rewards)/1319,"terminal_gsm8k_correct":int(sum(rewards)),"terminal_gsm8k_prompt_count":1319,"terminal_gsm8k_prompt_manifest_sha256":hashlib.sha256(prompts_path.read_bytes()).hexdigest()}}}}
result_path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n")
'''
    script = replace_block(script, 6, outcome)
    script = script.replace(
        "m4-oars-confirmatory-gsm8k-prompt-manifest-v1",
        "m4-oars-v2-quality-primary-gsm8k-prompt-manifest-v1",
    )
    script = re.sub(
        r"echo M4_OARS_CONFIRMATORY_[A-Z0-9_]+_CAPTURE_COMPLETE_OUTCOME_EMBARGOED",
        (
            "echo M4_OARS_V2_QUALITY_PRIMARY_"
            + run["identity"].replace("-", "_").upper()
            + "_CAPTURE_COMPLETE_OUTCOME_EMBARGOED"
        ),
        script,
        count=1,
    )
    forbidden = (
        "opportunity_at_risk_shadow.",
        "m4_oars_actuation_qualification.py",
        "m4-oars-randomized-confirmatory-arm-result-v1",
        "3bc49412a212419635e55b616e080708e17cf265c1926c1d93faebf072a87bfd",
        "90dbb632026591f24c966a43edd05b1e4933358b",
    )
    if any(token in script for token in forbidden):
        raise RuntimeError("historical scientific surface remains in candidate")

    candidate["spec"]["name"] = f"{stem}-acquisition"
    candidate["spec"]["script"] = script
    candidate["spec"]["time_limit"] = 14400
    candidate["spec"].pop("sbatch_additional_flags", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, separators=(",", ":")) + "\n")
    print(args.output, sha256(args.output))


if __name__ == "__main__":
    main()

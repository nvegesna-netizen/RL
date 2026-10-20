"""Close capture artifacts without evaluating either post-update replay arm."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from research.m4_eligible_replay.contracts import canonical_json, digest_file
from research.m4_paired_replay.capture_completion import prove_full_window

DOMAIN = "m4-eligible-live-causal-acquisition-v2"


def finish(
    *,
    assets: Path,
    identity: str,
    seed: int,
    run_exit_code: int,
    protocol: Path,
    analyzer: Path,
) -> dict:
    """Require an independent analyzer or a complete, zero-eligible window.

    Exceptions propagate. The shell records a failed job; it must not silently
    convert missing, partial, or invalid evidence into a no-event observation.
    """
    capsule = assets / "m4-eligible-live-capsule"
    prefix = f"m4-eligible-live-{identity}"
    watchdog = assets / f"{prefix}-watchdog.json"
    if run_exit_code == 0 and (capsule / "manifest.json").is_file():
        if watchdog.exists():
            raise ValueError("watchdog record present alongside capsule")
        # The qualified independent analyzer needs Torch only on the capsule path.
        spec = importlib.util.spec_from_file_location("capture_analyzer", analyzer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.analyze(capsule, protocol)
        if result["status"] != "PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE" or not all(
            value is True for value in result["checks"].values()
        ):
            raise ValueError("independent capture analyzer failed")
    else:
        proof = prove_full_window(
            run_exit_code=run_exit_code,
            run_log=(assets / f"{prefix}-run.log").read_bytes(),
            lifecycle=(assets / f"{prefix}-lifecycle.jsonl").read_bytes(),
            oars=(assets / f"{prefix}-oars-v2.jsonl").read_bytes(),
            capsule_written=capsule.exists(),
            watchdog_record_present=watchdog.exists(),
        )
        if proof["eligible_divergence_count"] != 0:
            raise ValueError("eligible divergence without a captured capsule")
        with (assets / f"{prefix}-progress.json").open("xb") as stream:
            stream.write(canonical_json(proof))
        result = {
            "schema": "m4-shield-eligible-live-acquisition-capture-result-v1",
            "status": "NO_ELIGIBLE_DECISION",
            "trajectory_identity": identity,
            "trajectory_seed": seed,
            "assignment_domain": DOMAIN,
            "maximum_completed_updates": 64,
            "run_exit_code": run_exit_code,
            "post_update_outcomes_opened": False,
            "paired_replay_started": False,
        }
    if (
        result["trajectory_identity"] != identity
        or result["trajectory_seed"] != seed
        or result["assignment_domain"] != DOMAIN
        or result["post_update_outcomes_opened"] is not False
    ):
        raise ValueError("capture assignment or outcome boundary differs")
    with (assets / f"{prefix}-capture-result.json").open("xb") as stream:
        stream.write(canonical_json(result))
    return result


def inventory(assets: Path, identity: str) -> None:
    """Inventory closed owned artifacts, excluding mutable JET wrapper logs."""
    prefix = f"m4-eligible-live-{identity}-"
    files = sorted(
        path
        for path in assets.iterdir()
        if path.name.startswith(prefix) and path.is_file()
    )
    capsule = assets / "m4-eligible-live-capsule"
    if capsule.exists():
        files += sorted(path for path in capsule.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in assets.rglob("*")):
        raise ValueError("capture assets contain symlink")
    with (assets / "artifacts.sha256").open("x") as stream:
        for path in files:
            stream.write(f"{digest_file(path)}  {path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-exit-code", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = finish(
            assets=args.assets,
            identity=args.identity,
            seed=args.seed,
            run_exit_code=args.run_exit_code,
            protocol=args.protocol,
            analyzer=args.analyzer,
        )
    finally:
        inventory(args.assets, args.identity)
    print(result["status"], args.identity, args.seed)


if __name__ == "__main__":
    main()

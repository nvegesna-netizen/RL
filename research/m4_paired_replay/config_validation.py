"""Execute native cross-section config constraints on CPU, without the GPU stack.

This is an offline verification tool, not a replacement for native MasterConfig.
It executes the original async/sampler/loss models and the complete unmodified
cross-section validator. Native GRPO defaults and policy mappings come from the
actual resolved YAML. EOS additionally constructs the full native MasterConfig.
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, NonNegativeInt, model_validator


def definitions(root: Path) -> dict:
    """Load only original declarative definitions and CPU validation functions."""
    namespace = {
        "__name__": "m4_native_config_offline",
        "BaseModel": BaseModel,
        "Field": Field,
        "NonNegativeInt": NonNegativeInt,
        "model_validator": model_validator,
        "Annotated": Annotated,
        "Any": Any,
        "Literal": Literal,
        "Optional": Optional,
        "Union": Union,
        "Path": Path,
        "math": math,
        "_DIGEST_CARDINALITY": 1 << 256,
    }
    selected = (
        (
            "nemo_rl/algorithms/async_utils/controlled_release.py",
            {
                "ControlledReleaseArmConfig",
                "_default_arms",
                "ControlledReleaseDelayConfig",
            },
        ),
        (
            "nemo_rl/algorithms/async_utils/staleness_sampler.py",
            {
                "WindowedSamplerConfig",
                "WeightFifoSamplerConfig",
                "InOrderSamplerConfig",
                "CustomSamplerConfig",
                "SamplerConfig",
                "_gated_required_buffer_capacity",
                "required_buffer_capacity_for_config",
            },
        ),
        ("nemo_rl/algorithms/loss/loss_functions.py", {"ClippedPGLossConfig"}),
        ("nemo_rl/algorithms/advantage_estimator.py", {"AdvEstimatorConfig"}),
        ("nemo_rl/algorithms/reward_functions.py", {"RewardShapingConfig"}),
        (
            "nemo_rl/algorithms/grpo.py",
            {"RewardScalingConfig", "AsyncGRPOConfig", "GRPOConfig"},
        ),
    )
    source = root / "nemo_rl/algorithms/single_controller_utils/config.py"
    config_tree = ast.parse(source.read_bytes())
    config_names = {
        node.name
        for node in config_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name not in {"MasterConfig", "AdvantageConfig"}
    } | {"validate_single_controller_config", "validate_sampler_buffer_capacity"}
    for relative, names in (*selected, (str(source.relative_to(root)), config_names)):
        tree = ast.parse((root / relative).read_bytes())
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", None) in names
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        ]
        if len(nodes) != len(names):
            raise ValueError(f"native definition selection incomplete: {relative}")
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    for value in tuple(namespace.values()):
        if (
            isinstance(value, type)
            and issubclass(value, BaseModel)
            and value is not BaseModel
        ):
            value.model_rebuild(_types_namespace=namespace)
    return namespace


def validate_native_constraints(root: Path, raw: dict) -> None:
    """Execute the complete native validator, with genuine nested native models."""
    native = definitions(root)
    grpo = native["GRPOConfig"](**raw["grpo"])
    config = SimpleNamespace(
        async_rl=native["AsyncRLConfig"](**raw["async_rl"]),
        grpo=grpo,
        policy=raw["policy"],
        env=raw["env"],
        loss_fn=native["ClippedPGLossConfig"](**raw["loss_fn"]),
    )
    native["validate_single_controller_config"](config)


def execute_capture_overrides(
    runtime: str, raw: dict, *, identity: str, seed: int, watchdog: str, domain: str
) -> dict:
    """Execute the actual packaged override statements, not a hand-copied fixture."""
    block = runtime.split("identity=sys.argv[3]; seed=int(sys.argv[4])", 1)[1]
    block = block.split("raw=OmegaConf.to_container(", 1)[1].split("\n", 1)[1]
    block = block.split("config=MasterConfig(", 1)[0]
    block = block.replace("@ASSIGNMENT_DOMAIN@", domain)
    namespace = {
        "raw": copy.deepcopy(raw),
        "identity": identity,
        "seed": seed,
        "sys": SimpleNamespace(
            argv=["-", "plan", "lock", identity, str(seed), watchdog]
        ),
    }
    exec(compile(block, "packaged-native-config-overrides", "exec"), namespace)
    return namespace["raw"]

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

import hashlib
import json

import pytest
import torch

from nemo_rl.algorithms.async_utils.gradient_utility_audit import (
    GradientUtilityAuditRecorder,
    hash_model_parameters,
    summarize_open_step_gradients,
)


def _parameter(values: list[float]) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(len(values)))
    parameter.grad = torch.tensor(values)
    return parameter


def test_summary_is_normalized_deterministic_and_nonmutating():
    parameter = _parameter([3.0, 4.0, 0.0, -12.0])
    before = parameter.grad.clone()

    first = summarize_open_step_gradients(
        [("weight", parameter)],
        normalization_tokens=2,
        sketch_bins=8,
        sketch_seeds=(11, 13),
        chunk_elements=2,
    )
    second = summarize_open_step_gradients(
        [("weight", parameter)],
        normalization_tokens=2,
        sketch_bins=8,
        sketch_seeds=(11, 13),
        chunk_elements=3,
    )

    assert first.exact_l2_norm == pytest.approx(6.5)
    assert first.coordinate_count == 4
    assert first.nonzero_coordinate_count == 3
    assert first.normalization_tokens == 2
    assert first.sketches == second.sketches
    assert first.sketches[0] != first.sketches[1]
    assert torch.equal(parameter.grad, before)


def test_summary_prefers_main_grad_and_rejects_nonfinite():
    parameter = _parameter([100.0, 100.0])
    parameter.main_grad = torch.tensor([5.0, 12.0])
    summary = summarize_open_step_gradients(
        [("weight", parameter)],
        normalization_tokens=1,
        sketch_bins=4,
        sketch_seeds=(1,),
    )
    assert summary.exact_l2_norm == pytest.approx(13.0)

    parameter.main_grad[0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_open_step_gradients(
            [("weight", parameter)],
            normalization_tokens=1,
            sketch_bins=4,
            sketch_seeds=(1,),
        )


def test_parameter_hash_is_byte_sensitive_and_stable():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    first = hash_model_parameters([("weight", parameter)], chunk_bytes=1)
    assert first == hash_model_parameters([("weight", parameter)], chunk_bytes=32)
    with torch.no_grad():
        parameter[1] = 3.0
    assert first != hash_model_parameters([("weight", parameter)])
    assert len(first) == hashlib.sha256().digest_size * 2


def test_recorder_emits_canonical_complete_ledger(tmp_path):
    recorder = GradientUtilityAuditRecorder(sketch_bins=2, sketch_seeds=(3, 5))
    recorder.append_group(
        audit_index=0,
        group_id="g0",
        sample_ids=("s0", "s1"),
        metadata={"gradient_opportunity_l1": 7.0},
        summary={"exact_l2_norm": 2.0, "sketches": [[1.0, -1.0]]},
    )
    recorder.append_terminal(
        parameter_sha256_before="a" * 64,
        parameter_sha256_after="a" * 64,
        groups=1,
        finish_train_step_calls=0,
    )
    output = tmp_path / "audit.jsonl"
    recorder.flush_jsonl(str(output))
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "header",
        "group_gradient",
        "terminal",
    ]
    assert events[1]["fold"] == 0
    assert events[2]["parameter_hash_unchanged"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"normalization_tokens": 0}, "normalization_tokens"),
        ({"sketch_bins": 1}, "sketch_bins"),
        ({"sketch_seeds": (1, 1)}, "sketch_seeds"),
    ],
)
def test_summary_rejects_invalid_contract(kwargs, message):
    arguments = {
        "normalization_tokens": 1,
        "sketch_bins": 4,
        "sketch_seeds": (1,),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        summarize_open_step_gradients(
            [("weight", _parameter([1.0]))],
            **arguments,
        )

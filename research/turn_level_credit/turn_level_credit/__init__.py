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

"""Turn-level credit assignment research utilities."""

from turn_level_credit.advantage import TurnLevelGRPOAdvantageEstimator
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.trace import TurnBatch
from turn_level_credit.verifier_credit import (
    VerifierCreditTransformConfig,
    VerifierScoreBatch,
    compute_verifier_credit,
    normalize_credit_within_prompt_turn,
    postprocess_verifier_credit,
)

__all__ = [
    "TurnBatch",
    "TurnCreditConfig",
    "TurnLevelGRPOAdvantageEstimator",
    "VerifierCreditTransformConfig",
    "VerifierScoreBatch",
    "compute_verifier_credit",
    "normalize_credit_within_prompt_turn",
    "postprocess_verifier_credit",
]

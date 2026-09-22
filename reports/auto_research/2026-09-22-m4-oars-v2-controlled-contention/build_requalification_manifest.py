# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the authorized repaired controlled-frontier requalification."""

from __future__ import annotations

import build_qualification_manifest as base

base.SOURCE_COMMIT = "2f017505f90d428dbc968aad7d718826cecb524c"
base.AUTHORIZATION_SHA256 = (
    "bc5357d27f4cb0530bc2282f8a4f992a5e256802d2b3e16df279f71fa4b73bd2"
)
base.LOCAL_VERIFICATION_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-controlled-contention/"
    "repair_verification.json"
)
base.LOCAL_VERIFICATION_SHA256 = (
    "176b9db68428f767d94a60e10e0455f65800da66cfac4914303f7570d3069c13"
)
base.NAME = "m4-oars-v2-controlled-frontier-requalification"


if __name__ == "__main__":
    base.main()

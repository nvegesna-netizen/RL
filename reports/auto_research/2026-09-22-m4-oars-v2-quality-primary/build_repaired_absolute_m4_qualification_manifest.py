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

"""Build absolute-M4 qualification from the repaired analyzer source."""

from __future__ import annotations

import build_actuation_qualification_manifest as base

base.SOURCE_COMMIT = "50443383825adeddca9cd1417c9dd261d8413670"
base.SOURCE_SHA256 = "3bab95936ff2e419532f3816ea96a53c3917b4716bccd460885d2606725d5394"
base.ANALYZER_SHA256 = (
    "856584b209d7ab0dd0fb4771c93e3234004e5fbe951f34f3c3c39ac446cff9c1"
)
base.VERIFICATION_PATH = (
    "reports/auto_research/2026-09-22-m4-oars-v2-quality-primary/"
    "actuation_analyzer_repair_verification.json"
)
base.VERIFICATION_SHA256 = (
    "318f1e62e0dde79d79ed17c950ecf6588d00b1e00908a81a8c242f61bb9ff6bc"
)


if __name__ == "__main__":
    base.main()

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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_rl.utils.terminal_policy_export import export_terminal_policy


class TerminalPolicyExportTest(unittest.TestCase):
    def _config(self, steps: int = 7):
        return SimpleNamespace(
            grpo=SimpleNamespace(max_num_steps=steps),
            model_dump=MagicMock(return_value={"grpo": {"max_num_steps": steps}}),
        )

    def test_completed_export_is_atomic_and_non_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "terminal-policy"

            def save_checkpoint(*, weights_path, optimizer_path, tokenizer_path):
                self.assertIsNone(optimizer_path)
                self.assertIsNone(tokenizer_path)
                Path(weights_path).mkdir(parents=True)

            trainer = SimpleNamespace(
                save_checkpoint=MagicMock(side_effect=save_checkpoint),
                finalize_async_save=MagicMock(),
            )
            manifest = export_terminal_policy(
                trainer=trainer,
                master_config=self._config(),
                output_dir=str(destination),
                train_steps=7,
                trainer_version=7,
            )

            self.assertEqual(manifest["train_steps"], 7)
            self.assertFalse(manifest["resumable_training_checkpoint"])
            self.assertTrue((destination / "policy" / "weights").is_dir())
            self.assertTrue((destination / "resolved_config.json").is_file())
            on_disk = json.loads(
                (destination / "terminal_policy_export.json").read_text()
            )
            self.assertEqual(on_disk, manifest)
            self.assertFalse((Path(tmp) / ".terminal-policy.incomplete").exists())
            trainer.finalize_async_save.assert_called_once_with()

    def test_incomplete_boundary_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "terminal-policy"
            with self.assertRaisesRegex(RuntimeError, "requires.*boundary"):
                export_terminal_policy(
                    trainer=MagicMock(),
                    master_config=self._config(),
                    output_dir=str(destination),
                    train_steps=6,
                    trainer_version=6,
                )
            self.assertFalse(destination.exists())

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "terminal-policy"
            destination.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refuses to overwrite"):
                export_terminal_policy(
                    trainer=MagicMock(),
                    master_config=self._config(),
                    output_dir=str(destination),
                    train_steps=7,
                    trainer_version=7,
                )


if __name__ == "__main__":
    unittest.main()

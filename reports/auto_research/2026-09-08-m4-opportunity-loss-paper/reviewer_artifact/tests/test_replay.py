import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReviewerArtifactTest(unittest.TestCase):
    def test_offline_replay(self) -> None:
        result = subprocess.run(
            [sys.executable, "analysis/replay.py", "--verify"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "PASS")
        self.assertEqual(output["published"]["full_window_assignments"], 106_653)
        self.assertEqual(output["published"]["llama_1b_extension_assignments"], 28_712)
        self.assertEqual(output["published"]["llama_3b_extension_assignments"], 28_788)
        self.assertEqual(output["published"]["downstream_quality_blocks"], 16)
        self.assertEqual(output["published"]["downstream_quality_runs"], 32)
        self.assertEqual(output["published"]["oars_pairs"], 10)
        self.assertEqual(output["published"]["oars_runs"], 20)
        self.assertEqual(output["published"]["quality_primary_blocks"], 18)
        self.assertEqual(output["published"]["quality_primary_runs"], 54)
        self.assertEqual(output["published"]["m4_shield_decisions"], 64)
        self.assertEqual(output["published"]["m4_shield_interventions"], 5)
        self.assertEqual(output["published"]["gradient_utility_groups"], 256)
        self.assertEqual(output["published"]["gradient_utility_primary_groups"], 214)
        self.assertEqual(output["synthetic"]["row_count"], 192)

    def test_figure_reproduction(self) -> None:
        result = subprocess.run(
            [sys.executable, "analysis/render_figures.py", "--verify"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("FIGURE_RENDER_VERIFY_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()

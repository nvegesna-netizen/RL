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
        self.assertEqual(output["published"]["full_window_assignments"], 77_865)
        self.assertEqual(output["published"]["llama_extension_assignments"], 28_712)
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

"""Fresh-process import-path regression, without Torch/Ray initialization."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BootstrapTests(unittest.TestCase):
    def test_complete_bridge_and_core_namespace_in_fresh_process(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = """import importlib.util,sys
assert 'nemo_rl' not in sys.modules
from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron
core=initialize_bundled_megatron(require_bridge=True)
bridge=core.parent.parent/'src'
assert importlib.util.find_spec('megatron.core').origin==str(core/'megatron/core/__init__.py')
assert importlib.util.find_spec('megatron.bridge').origin==str(bridge/'megatron/bridge/__init__.py')
before=list(sys.path); initialize_bundled_megatron(require_bridge=True); assert before==sys.path
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fresh_process_finds_exact_bundled_sources(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = """import importlib.util,json,sys
assert 'nemo_rl' not in sys.modules
from research.m4_eligible_replay.bootstrap import initialize_bundled_megatron
core=initialize_bundled_megatron()
assert str(core) in sys.path
assert importlib.util.find_spec('megatron.core').origin==str(core/'megatron/core/__init__.py')
before=list(sys.path); initialize_bundled_megatron(); assert before==sys.path
assert 'nemo_rl' not in sys.modules
print(json.dumps({'fresh_process':True,'bundled_modules_resolved':True}))
"""
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)["bundled_modules_resolved"])

    def test_worker_bootstrap_precedes_megatron_imports(self) -> None:
        path = Path(__file__).with_name("worker.py")
        tree = ast.parse(path.read_bytes())
        call = next(
            node.lineno
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "initialize_bundled_megatron"
        )
        imports = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("megatron")
        ]
        self.assertTrue(imports)
        self.assertLess(call, min(imports))


if __name__ == "__main__":
    unittest.main()

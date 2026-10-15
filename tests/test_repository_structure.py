"""Guard maintained entry points against reintroducing retired dependencies."""

import ast
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import COMMANDS, main

ROOT = Path(__file__).resolve().parents[1]
RETIRED = (
    "src.experiments", "src.analysis", "src.split", "src.train", "src.inference", "src.utils",
    "utils.features", "utils.folds", "utils.metrics", "utils.plotting", "utils.tabpfn",
    "utils.target", "utils.validation", "tabpfn_extensions",
)


class RepositoryStructureTests(unittest.TestCase):
    def test_all_commands_import_and_expose_main(self):
        for command, (module, _) in COMMANDS.items():
            with self.subTest(command=command):
                self.assertTrue(callable(getattr(importlib.import_module(module), "main", None)))

    def test_dispatch_preserves_arguments_and_restores_argv(self):
        original = sys.argv
        captured = []
        with patch("src.cli.runpy.run_module", side_effect=lambda *a, **k: captured.append(list(sys.argv))):
            main(["interpolate-fly", "--input-dir", "data/example"])
        self.assertEqual(captured, [["src.regression.fly_tabpfn", "--input-dir", "data/example"]])
        self.assertIs(sys.argv, original)

    def test_failed_dispatch_also_restores_argv(self):
        original = sys.argv
        with patch("src.cli.runpy.run_module", side_effect=RuntimeError("example")):
            with self.assertRaises(RuntimeError):
                main(["fd-transfer-ensemble"])
        self.assertIs(sys.argv, original)

    def test_no_maintained_code_imports_retired_modules(self):
        for directory in ("src", "utils", "tests", "web-demo"):
            for path in (ROOT / directory).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        name = node.module or ""
                        if node.level:
                            package = ".".join(path.relative_to(ROOT).parts[:-1])
                            name = importlib.util.resolve_name("." * node.level + name, package)
                        names = [name, *(name + "." + alias.name for alias in node.names)]
                    else:
                        continue
                    for name in names:
                        with self.subTest(path=path, module=name):
                            self.assertFalse(any(name == old or name.startswith(old + ".") for old in RETIRED))


if __name__ == "__main__":
    unittest.main()

"""Check the core source package without data, settings, Torch or a GPU."""

from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_local_imports(files):
    """Check local module/member references without importing Torch or TabPFN."""
    modules = {}
    for path in files:
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[0] == "stage_models":
            parts.pop(0)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for name, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.split(".")[0] not in {
                "ptb_tabpfn",
                "experiments",
                "analysis",
            }:
                continue
            assert node.module in modules, (name, node.module)
            target = modules[node.module]
            available = {
                n.name
                for n in target.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            }
            for item in target.body:
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    available.update(
                        alias.asname or alias.name.split(".")[0] for alias in item.names
                    )
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        item.targets if isinstance(item, ast.Assign) else [item.target]
                    )
                    available.update(
                        n.id
                        for t in targets
                        for n in ast.walk(t)
                        if isinstance(n, ast.Name)
                    )
            for alias in node.names:
                assert (
                    alias.name in available or f"{node.module}.{alias.name}" in modules
                ), (name, node.module, alias.name)
                count += 1
    return count


def main():
    files = [
        p
        for p in ROOT.rglob("*.py")
        if not {".git", "__pycache__", "outputs"} & set(p.relative_to(ROOT).parts)
    ]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = check_local_imports(files)
    for name in ("configs", "experiments", "analysis"):
        assert not (ROOT / name).exists(), f"Excluded release component remains: {name}"
    assert list(ROOT.rglob("README.md")) == [ROOT / "README.md"]
    print(
        f"PASS: {len(files)} Python files; {count} local import references; core-only release; no bundled study settings."
    )


if __name__ == "__main__":
    main()

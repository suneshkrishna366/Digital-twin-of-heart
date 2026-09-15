import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path("cell-models/cell_models")

# Python standard-library modules
stdlib = set(getattr(sys, "stdlib_module_names", set()))

imports = set()

for py_file in ROOT.rglob("*.py"):
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

print("Checking imported modules...\n")

missing = []

for module in sorted(imports):
    if module in stdlib:
        continue

    # Local package
    if module == "cell_models":
        continue

    try:
        found = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        found = None

    if found is None:
        missing.append(module)
        print(f"❌ MISSING: {module}")
    else:
        print(f"✅ FOUND:   {module}")

print("\n" + "=" * 50)

if missing:
    print("Missing third-party modules:")
    for module in missing:
        print(f"  - {module}")

    print("\nInstall them with:")
    print(
        f'{sys.executable} -m pip install '
        + " ".join(missing)
    )
else:
    print("🎉 No missing modules detected!")
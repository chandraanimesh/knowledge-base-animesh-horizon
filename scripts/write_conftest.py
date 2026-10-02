from pathlib import Path

SKIP = ["test_auth_output.txt", "test_errors.txt", "test_errors2.txt", "test_errors3.txt"]

lines = [
    "# Narrowly scoped: skip only specific binary .txt files that crash pytest.",
    "_SKIP_FILES = {",
]
for name in SKIP:
    lines.append(f'    "{name}",')
lines += [
    "}",
    "",
    "",
    "def pytest_ignore_collect(collection_path, config):",
    '    """Skip known binary output files in the project root."""',
    "    return collection_path.name in _SKIP_FILES",
    "",
]

target = Path("tests") / "conftest.py"
target.write_text("\n".join(lines), encoding="utf-8")
print(f"Written {target}")

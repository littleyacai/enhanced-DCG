"""Audit repository structure, portability, integrity, and Python syntax."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "LICENSE-DATA.md",
    "CITATION.cff",
    "requirements.txt",
    "Geoborehole.py",
    "data/example/SFGC-BX-007-III-1-9_image.avi",
    "data/example/SFGC-BX-007-III-1-9.xlsx",
    "step/lib/image/logs/ep300-loss0.031-val_loss0.053.pth",
]
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
HAN = re.compile(r"[\u3400-\u9fff]")
MAX_GITHUB_FILE = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = []
    warnings = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            failures.append(f"syntax error in {relative}: {exc}")
        if DRIVE_PATH.search(text):
            failures.append(f"hard-coded drive path in {relative}")
        try:
            with path.open("rb") as stream:
                for token in tokenize.tokenize(stream.readline):
                    if token.type == tokenize.COMMENT and HAN.search(token.string):
                        failures.append(f"non-English comment in {relative}:{token.start[0]}")
        except tokenize.TokenError as exc:
            failures.append(f"tokenization error in {relative}: {exc}")

    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for path in files:
        if path.stat().st_size >= MAX_GITHUB_FILE:
            failures.append(f"file reaches GitHub's 100 MiB limit: {path.relative_to(ROOT)}")
    if sum(path.stat().st_size for path in files) > 1024**3:
        warnings.append("repository exceeds 1 GiB; use a data archive or Git LFS")

    manifest = ROOT / "MANIFEST.sha256"
    if manifest.is_file():
        expected = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                checksum, relative = line.split("  ", 1)
                expected[relative] = checksum
        for relative, checksum in expected.items():
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"manifest entry is missing: {relative}")
            elif sha256(path) != checksum:
                failures.append(f"checksum mismatch: {relative}")
    else:
        warnings.append("MANIFEST.sha256 is absent")

    print(f"Repository: {ROOT}")
    print(f"Files: {len(files)}; Python files: {len(python_files)}")
    print(f"Size: {sum(path.stat().st_size for path in files) / 1024**2:.2f} MiB")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"Repository audit failed with {len(failures)} issue(s).")
        return 1
    print("Repository audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

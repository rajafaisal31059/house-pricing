import io
import os
import sys
from pathlib import Path
import tokenize

EXCLUDE_DIRS = {'.git', 'venv', '.idea', '.pytest_cache', '__pycache__'}


def strip_python_comments(path: Path) -> bool:
    """Remove # comments from a Python file, preserving code and docstrings.
    Returns True if file was modified."""
    try:
        with tokenize.open(path) as f:
            src = f.read()
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        new_tokens = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        new_src = tokenize.untokenize(new_tokens)
        if new_src != src:
            path.write_text(new_src, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"SKIP {path}: {e}")
        return False


def strip_gitignore_comments(path: Path) -> bool:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        kept = [ln for ln in lines if not ln.lstrip().startswith('#')]
        new = "\n".join(kept) + ("\n" if kept else "")
        old = "\n".join(lines) + ("\n" if lines else "")
        if new != old:
            path.write_text(new, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"SKIP {path}: {e}")
        return False


def should_skip(p: Path) -> bool:
    parts = set(p.parts)
    return any(d in EXCLUDE_DIRS for d in parts)


def main(root: Path) -> int:
    changed = []
    for path in root.rglob('*.py'):
        if should_skip(path):
            continue
        if strip_python_comments(path):
            changed.append(str(path.relative_to(root)))
    gi = root / '.gitignore'
    if gi.exists() and strip_gitignore_comments(gi):
        changed.append('.gitignore')
    print('Modified files:')
    for c in changed:
        print(' -', c)
    print(f'Total modified: {len(changed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(__file__).resolve().parents[1])) 
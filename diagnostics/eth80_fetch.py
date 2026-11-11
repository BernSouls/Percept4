"""Fetch ONLY the ETH-80 images named in eth80_split.json (154 files, ~16 MB).

The dataset itself is not redistributed with this repository: it is pulled
from a public GitHub mirror with a blob-less partial clone and a sparse
checkout, so nothing outside the split is downloaded.

    python diagnostics/eth80_fetch.py            # -> data/ETH-80/<class>/<file>.png

`data/` is git-ignored.  Re-running is idempotent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eth80_split import CLASS_DIR, load_split  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "ETH-80")


def split_files(split=None):
    """[(class, repo-relative path)] for every image the split names."""
    split = split or load_split()
    out = [(cls, p) for cls, p in split["references"].items()]
    for name in ("eval", "heldout"):
        for cls, paths in split[name].items():
            out.extend((cls, p) for p in paths)
    return out


def local_path(cls: str, rel: str) -> str:
    return os.path.join(DATA_DIR, cls, os.path.basename(rel))


def _run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    split = load_split()
    files = split_files(split)
    missing = [(c, p) for c, p in files if not os.path.isfile(local_path(c, p))]
    if not missing:
        print("ETH-80 subset already present: %d files in %s" % (len(files), DATA_DIR))
        return 0
    url = "https://github.com/%s.git" % split["dataset_repo"]
    tmp = tempfile.mkdtemp(prefix="eth80_")
    try:
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet",
              url, tmp])
        _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=tmp)
        patterns = "\n".join("/" + p for _c, p in missing) + "\n"
        with open(os.path.join(tmp, ".git", "info", "sparse-checkout"), "w") as fh:
            fh.write(patterns)
        _run(["git", "checkout", "--quiet", split["dataset_commit"]], cwd=tmp)
        for cls, rel in missing:
            src = os.path.join(tmp, rel)
            if not os.path.isfile(src):
                raise SystemExit("not in dataset checkout: %s" % rel)
            dst = local_path(cls, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("fetched %d files into %s" % (len(missing), DATA_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

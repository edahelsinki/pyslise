"""Generate the code reference pages and navigation."""

from pathlib import Path
from typing import List

import mkdocs_gen_files

DOCS = Path("docs")

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("slise").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.parent.with_suffix(".md")

    ident = ".".join(parts)
    nav[ident] = doc_path.as_posix()

    with mkdocs_gen_files.open(DOCS / doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n")
        fd.write(f"::: {ident}")

    # mkdocs_gen_files.set_edit_path(doc_path, path)

with mkdocs_gen_files.open(DOCS / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

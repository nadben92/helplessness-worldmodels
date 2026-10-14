"""Checks paper/sections/*.tex and paper/reports/*.tex and reports, with file and line:

  1. `% src: <path>` comments whose path (relative to the repo root) does not exist; the keyword
     `% src: notes` stands for the author's notes (SRC_KEYWORDS), which must exist too;
  2. numbers written by hand in the text, which should come from numbers.tex;
  3. remaining \\todo{} notes.

Rules for (2): every digit outside comments is reported, in prose and in
math alike; math without digits is never reported. Not reported: digits in
the arguments of the commands in IGNORED_ARG_COMMANDS (\\label, \\ref,
\\cite..., \\includegraphics, \\vspace...), in the content of \\todo{}, and
in optional [...] arguments, and digits glued to a preceding letter
(identifiers such as E1 or L1). A line ending with `% num-ok` is skipped. In a
file with a preamble (a report), only the body after \begin{document} is
checked for numbers: the preamble is layout, not prose.

Exit code 1 if (1) or (2) finds anything; \\todo notes are only listed.

Usage: python paper/check_sources.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
SECTIONS_DIR = PAPER_DIR / "sections"
REPORTS_DIR = PAPER_DIR / "reports"
# `% src: <keyword>` -> the file it stands for (relative to the repo root).
SRC_KEYWORDS = {"notes": "paper/sections/e3_notes_fr.md"}
BEGIN_DOCUMENT = re.compile(r"\\begin\{document\}")

NUM_OK_MARKER = "% num-ok"
IGNORED_ARG_COMMANDS = (
    "label", "ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref",
    "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "nocite",
    "includegraphics", "input", "include", "begin", "end", "url", "href", "hyperref",
    "vspace", "hspace", "setlength", "addtolength", "todo",
)

SRC_COMMENT = re.compile(r"%\s*src:\s*(\S+)")
TODO = re.compile(r"\\todo\b")
UNESCAPED_PERCENT = re.compile(r"(?<!\\)%")
IGNORED_ARGS = re.compile(
    r"\\(?:" + "|".join(IGNORED_ARG_COMMANDS) + r")\*?\s*(?:\[[^\]]*\]\s*)*(?:\{[^{}]*\}\s*)?"
)
OPTIONAL_ARG = re.compile(r"\[[^\]]*\]")
COMMAND_NAME = re.compile(r"\\[A-Za-z@]+")
# A digit glued to a preceding letter is part of an identifier (E1, L1), not a number.
NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)*")


def strip_comment(line: str) -> str:
    m = UNESCAPED_PERCENT.search(line)
    return line if m is None else line[:m.start()]


def hard_numbers(line: str) -> list[str]:
    if line.rstrip().endswith(NUM_OK_MARKER):
        return []
    text = strip_comment(line)
    text = IGNORED_ARGS.sub(" ", text)
    text = OPTIONAL_ARG.sub(" ", text)
    text = COMMAND_NAME.sub(" ", text)  # macro names never contain digits, but keep the text clean
    return NUMBER.findall(text)


def check_file(path: Path) -> tuple[list[str], list[str], list[str]]:
    missing, numbers, todos = [], [], []
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    has_preamble = any(BEGIN_DOCUMENT.search(strip_comment(line)) for line in lines)
    in_body = not has_preamble
    for lineno, line in enumerate(lines, start=1):
        where = f"{rel}:{lineno}"
        for m in SRC_COMMENT.finditer(line):
            src = SRC_KEYWORDS.get(m.group(1), m.group(1))
            if not (REPO_ROOT / src).exists():
                missing.append(f"{where}: src path does not exist: {src}")
        if not in_body:
            in_body = bool(BEGIN_DOCUMENT.search(strip_comment(line)))
            found = []
        else:
            found = hard_numbers(line)
        if found:
            numbers.append(f"{where}: {', '.join(found)}  |  {line.strip()}")
        if TODO.search(strip_comment(line)):
            todos.append(f"{where}: {line.strip()}")
    return missing, numbers, todos


def main() -> None:
    files = sorted(SECTIONS_DIR.glob("*.tex")) + sorted(REPORTS_DIR.glob("*.tex"))
    if not files:
        raise SystemExit(f"No .tex file in {SECTIONS_DIR} or {REPORTS_DIR}")
    missing, numbers, todos = [], [], []
    for path in files:
        m, n, t = check_file(path)
        missing += m
        numbers += n
        todos += t

    for title, items in (
        ("Missing % src: paths", missing),
        ("Hard-coded numbers (use a numbers.tex macro, or end the line with % num-ok)", numbers),
        ("Remaining \\todo", todos),
    ):
        print(f"== {title}: {len(items)}")
        for item in items:
            print(f"  {item}")

    sys.exit(1 if (missing or numbers) else 0)


if __name__ == "__main__":
    main()

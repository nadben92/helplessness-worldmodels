"""Writes paper/numbers.tex: one \\newcommand per number quoted in the paper or the E3 report.

Each number is declared once (registry.Number): its macro name, how it is read from a source, the source (written as a
`% src:` comment above the macro) and its display format. The E2 entries are below; the E1 and E3 reports' are in
e1_numbers.py and e3_numbers.py. Rounding happens in exactly one place, `fmt()`. A missing file, a missing row or key, or an
inconsistent value stops the script with an explicit message; nothing is ever defaulted.

The one exception is a number declared pending=True (the E3 final campaign): while its source does not exist, the
macro is \\pending{<name>} (macros.tex), and the script lists every such macro. A pending number whose source exists
but is inconsistent stops the script like any other.

Macro names contain no digits (a LaTeX constraint): experiment ids are spelled out, e.g. \\eTwoGapExponent for E2,
\\eThreeKs for E3.

Usage: python paper/numbers/gen_numbers.py
"""

from __future__ import annotations

import math
import re
import sys

from registry import PAPER_DIR, REPO_ROOT, Number, PendingSource, SourceError, constant_value, n_distinct, one_value

OUT_PATH = PAPER_DIR / "numbers.tex"

E2_CSV_DIR = "experiments/E2_online_4zones/artifacts/figures/csv"
E2_F2 = f"{E2_CSV_DIR}/F2_null_model_paired_diff.csv"
E2_TABLE3 = f"{E2_CSV_DIR}/table3_budget_trend_power_law.csv"
E2_TABLE1 = f"{E2_CSV_DIR}/table1_paired_diff_by_zone.csv"


# --- formatting (the only place where rounding happens) ----------------------

def _group(digits: str) -> str:
    """Thousands separators for integers of 5 digits or more (10{,}000; 1358 stays 1358)."""
    if len(digits) < 5:
        return digits
    head = len(digits) % 3 or 3
    return digits[:head] + "".join("{,}" + digits[i:i + 3] for i in range(head, len(digits), 3))


def fmt(value: float, decimals: int, signed: bool = False, style: str = "fixed") -> str:
    """LaTeX rendering of a number. style "fixed": `decimals` decimals (0: the value must be an integer);
    "rounded": `decimals` decimals, rounding allowed even at 0 (a value that is not an integer by nature, e.g. a fitted
    crossing); "percent": value x 100 with `decimals` decimals (no % sign: the text writes it); "sci": m x 10^e with
    `decimals` decimals on the mantissa (10^e alone when m = 1)."""
    if not math.isfinite(value):
        raise SourceError(f"Value {value!r} is not finite")
    if style == "sci":
        exponent = math.floor(math.log10(abs(value)))
        mantissa = f"{value / 10 ** exponent:.{decimals}f}"
        body = f"10^{{{exponent}}}" if float(mantissa) == 1.0 else f"{mantissa} \\times 10^{{{exponent}}}"
        return f"\\ensuremath{{{body}}}"
    if style == "percent":
        value = value * 100
    elif style not in ("fixed", "rounded"):
        raise SourceError(f"Unknown number style {style!r}")
    if decimals == 0 and style == "fixed":
        if abs(value - round(value)) > 1e-9:  # only float noise is tolerated (e.g. -1 / (1 - 0.9))
            raise SourceError(f"Value {value!r} declared as an integer is not one")
        text = str(int(round(value)))
    else:
        text = f"{value:.{decimals}f}"
    negative = text.startswith("-") and float(text) != 0.0
    text = text.lstrip("-")
    whole, _, frac = text.partition(".")
    text = _group(whole) + (f".{frac}" if frac else "")
    if negative:
        return f"\\ensuremath{{-{text}}}"
    return f"\\ensuremath{{+{text}}}" if signed and float(text.replace("{,}", "")) != 0.0 else text


# --- declarations ------------------------------------------------------------

E2_NUMBERS = [
    Number(
        "eTwoNSeeds",
        f"{E2_F2} :: n_seeds (constant over rows quantity=b, zone=controllable_noisy)",
        lambda: constant_value(E2_F2, "n_seeds", quantity="b", zone="controllable_noisy"),
        decimals=0,
    ),
    Number(
        "eTwoNCheckpoints",
        f"{E2_F2} :: number of distinct checkpoint values (quantity=b, zone=controllable_noisy)",
        lambda: n_distinct(E2_F2, "checkpoint", quantity="b", zone="controllable_noisy"),
        decimals=0,
    ),
    Number(
        "eTwoGapFitNPoints",
        f"{E2_TABLE3} :: n_points (quantity=b, zone=controllable_noisy)",
        lambda: one_value(E2_TABLE3, "n_points", quantity="b", zone="controllable_noisy"),
        decimals=0,
    ),
    Number(
        "eTwoGapExponent",
        f"{E2_TABLE3} :: exponent (quantity=b, zone=controllable_noisy)",
        lambda: one_value(E2_TABLE3, "exponent", quantity="b", zone="controllable_noisy"),
        decimals=3, signed=True,
    ),
    Number(
        "eTwoGapExponentCILow",
        f"{E2_TABLE3} :: exponent_bootstrap_ci95_low (quantity=b, zone=controllable_noisy)",
        lambda: one_value(E2_TABLE3, "exponent_bootstrap_ci95_low", quantity="b", zone="controllable_noisy"),
        decimals=3, signed=True,
    ),
    Number(
        "eTwoGapExponentCIHigh",
        f"{E2_TABLE3} :: exponent_bootstrap_ci95_high (quantity=b, zone=controllable_noisy)",
        lambda: one_value(E2_TABLE3, "exponent_bootstrap_ci95_high", quantity="b", zone="controllable_noisy"),
        decimals=3, signed=True,
    ),
    Number(
        "eTwoEffectCN",
        f"{E2_TABLE1} :: mean_diff (checkpoint=100000, zone=controllable_noisy, quantity=b): E2's online - matched gap "
        "on b in the CN zone, the effect E1 and E3 refer to",
        lambda: one_value(E2_TABLE1, "mean_diff", checkpoint="100000", zone="controllable_noisy", quantity="b"),
        decimals=2, signed=True,
    ),
]

MACRO_NAME = re.compile(r"^[A-Za-z]+$")


def all_numbers() -> list[Number]:
    import e1_numbers  # imported here: they read the E1 / E3 results and the experiments' modules
    import e3_numbers

    return E2_NUMBERS + e3_numbers.NUMBERS + e1_numbers.NUMBERS


def render(numbers: list[Number]) -> tuple[str, list[str]]:
    seen, pending = set(), []
    lines = [
        "% Generated by paper/numbers/gen_numbers.py -- do not edit by hand.",
        "% Rerun `python paper/build.py --numbers` after any change to the results.",
        "",
    ]
    for n in numbers:
        if not MACRO_NAME.match(n.macro):
            raise SourceError(f"Invalid macro name {n.macro!r}: letters only")
        if n.macro in seen:
            raise SourceError(f"Duplicate macro name {n.macro!r}")
        seen.add(n.macro)
        try:
            value = fmt(n.get(), n.decimals, n.signed, n.style)
        except PendingSource as e:
            if not n.pending:
                raise SourceError(f"{n.macro}: source pending ({e}) but the number is not declared pending")
            value = f"\\pending{{{n.macro}}}"
            pending.append(n.macro)
        lines.append(f"% src: {n.src}" + ("  [PENDING: the final campaign]" if n.pending else ""))
        lines.append(f"\\newcommand{{\\{n.macro}}}{{{value}}}")
    return "\n".join(lines) + "\n", pending


def main() -> None:
    numbers = all_numbers()
    text, pending = render(numbers)
    OUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(numbers)} numbers, {len(pending)} pending)")
    if pending:
        print("pending (written as \\pending{}): " + ", ".join(pending))


if __name__ == "__main__":
    sys.exit(main())

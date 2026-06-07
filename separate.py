#!/usr/bin/env python3
"""
Automatically detect the column position of a 2-column PDF and split it
into two separate text files (e.g. Euskera | Spanish).

This script first extracts layout-preserving text with pdftotext,
auto-detects the right column start position, then splits at that column.

Usage:
    python separate.py document.pdf
    python separate.py document.pdf --outdir ./output
    python separate.py document.pdf --eus-suffix -eus --es-suffix -es

Output:
    Creates <basename>-<eus_suffix>.txt and <basename>-<es_suffix>.txt
    in the same directory as the input (or in --outdir).
"""

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

PDFTOTEXT_CMD = ["pdftotext", "-layout", "-nopgbrk"]


def extract_text(pdf_path: str) -> str:
    """Run pdftotext -layout -nopgbrk and return extracted text."""
    try:
        result = subprocess.run(
            PDFTOTEXT_CMD + [pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError:
        sys.exit(
            "Error: 'pdftotext' not found. Install it with:\n"
            "  macOS:   brew install poppler\n"
            "  Ubuntu:  sudo apt install poppler-utils"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"pdftotext failed (exit code {e.returncode}):\n{e.stderr}")


def detect_column_position(text: str, min_pos: int = 50, sample_lines: int = 200) -> int:
    """Auto-detect the right column start position.

    Strategy: for each line, find the first non-space char after min_pos.
    The most frequent such position is the column boundary.
    """
    positions: list[int] = []
    for line in text.splitlines()[:sample_lines]:
        for i in range(min_pos, len(line)):
            if line[i] != " ":
                positions.append(i)
                break

    if not positions:
        sys.exit(
            "Could not detect column position. "
            "The text may not be 2-column layout. "
            "Try --col <position> to set it manually."
        )

    # Most frequent position is the column boundary
    counter = Counter(positions)
    most_common_pos, count = counter.most_common(1)[0]
    total = len(positions)
    pct = count / total * 100

    print(f"   → Detected column at position {most_common_pos} "
          f"({count}/{total} lines, {pct:.0f}%)")
    return most_common_pos


def split_columns(
    text: str,
    col: int,
) -> tuple[list[str], list[str]]:
    """Split text at fixed column position. Returns (left_lines, right_lines)."""
    left_lines: list[str] = []
    right_lines: list[str] = []
    for line in text.splitlines():
        if len(line) > col:
            left_lines.append(line[:col].rstrip())
            right_lines.append(line[col:].lstrip())
        else:
            left_lines.append(line.rstrip())
            right_lines.append("")
    return left_lines, right_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect column position and split a 2-column PDF "
                    "into two text files (e.g. Euskera | Spanish)."
    )
    parser.add_argument("pdf", help="Path to the 2-column PDF file")
    parser.add_argument(
        "--col",
        type=int,
        default=None,
        help="Manually set column position (skip auto-detection). Default: auto-detect",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Default: same directory as input PDF",
    )
    parser.add_argument(
        "--eus-suffix",
        type=str,
        default="-eus",
        help="Suffix for the left-column output file. Default: -eus",
    )
    parser.add_argument(
        "--es-suffix",
        type=str,
        default="-es",
        help="Suffix for the right-column output file. Default: -es",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"Expected a .pdf file, got '{pdf_path.suffix}'")

    outdir = Path(args.outdir) if args.outdir else pdf_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    eus_path = outdir / f"{stem}{args.eus_suffix}.txt"
    es_path = outdir / f"{stem}{args.es_suffix}.txt"

    # Step 1: Extract text
    print(f" Extracting text from: {pdf_path}")
    text = extract_text(str(pdf_path))
    total_lines = text.count("\n")
    print(f"   → {total_lines} lines extracted")

    # Step 2: Detect or set column position
    if args.col is not None:
        col = args.col
        print(f"   → Using manual column position: {col}")
    else:
        print("🔍 Auto-detecting column position...")
        col = detect_column_position(text)

    # Step 3: Split columns
    print(f"✂️  Splitting at column {col}...")
    left_lines, right_lines = split_columns(text, col=col)

    # Step 4: Write output files
    eus_path.write_text("\n".join(left_lines) + "\n", encoding="utf-8")
    es_path.write_text("\n".join(right_lines) + "\n", encoding="utf-8")

    print(f"   → {eus_path}  ({len(left_lines)} lines)")
    print(f"   → {es_path}  ({len(right_lines)} lines)")


if __name__ == "__main__":
    main()

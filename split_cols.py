#!/usr/bin/env python3
"""
Split a 2-column PDF into two separate text files (e.g. Euskera | Spanish).

Uses pdftotext to extract layout-preserving text, then splits each line
at a fixed column position into two output files.

Usage:
    python split_cols.py document.pdf
    python split_cols.py document.pdf --col 67
    python split_cols.py document.pdf --col 67 --outdir ./output
    python split_cols.py document.pdf --eus-suffix -eus --es-suffix -es

Output:
    Creates <basename>-<eus_suffix>.txt and <basename>-<es_suffix>.txt
    in the same directory as the input (or in --outdir).
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Shell command to extract layout-preserving text from PDF
PDFTOTEXT_CMD = ["pdftotext", "-layout", "-nopgbrk"]


def extract_text(pdf_path: str) -> str:
    """Run pdftotext -layout -nopgbrk on the PDF and return the extracted text."""
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
        description="Split a 2-column PDF into two text files (e.g. Euskera | Spanish)."
    )
    parser.add_argument("pdf", help="Path to the 2-column PDF file")
    parser.add_argument(
        "--col",
        type=int,
        default=67,
        help="Column position where the right column starts. Default: 67",
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

    # Determine output directory
    outdir = Path(args.outdir) if args.outdir else pdf_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Build output filenames
    stem = pdf_path.stem
    eus_path = outdir / f"{stem}{args.eus_suffix}.txt"
    es_path = outdir / f"{stem}{args.es_suffix}.txt"

    # Step 1: Extract text
    print(f" Extracting text from: {pdf_path}")
    text = extract_text(str(pdf_path))
    total_lines = text.count("\n")
    print(f"   → {total_lines} lines extracted")

    # Step 2: Split columns
    print(f"✂️  Splitting at column {args.col}...")
    left_lines, right_lines = split_columns(text, col=args.col)

    # Step 3: Write output files
    eus_path.write_text("\n".join(left_lines) + "\n", encoding="utf-8")
    es_path.write_text("\n".join(right_lines) + "\n", encoding="utf-8")

    print(f"   → {eus_path}  ({len(left_lines)} lines)")
    print(f"   → {es_path}  ({len(right_lines)} lines)")


if __name__ == "__main__":
    main()

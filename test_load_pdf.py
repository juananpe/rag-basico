"""
Test script for pdf_loader.load_pdf.

Usage:
    python test_load_pdf.py sample.pdf
"""

import argparse

from pdf_loader import load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a PDF into LangChain Document objects using pypdf."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1000,
        help="Number of characters to print per page. Default: 1000",
    )

    args = parser.parse_args()

    documents = load_pdf(args.pdf_path)

    print(f"Loaded {len(documents)} pages\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Page {index} ---")
        print(document.page_content[: args.preview_chars])
        print()
        print("Metadata:", document.metadata)
        print()


if __name__ == "__main__":
    main()

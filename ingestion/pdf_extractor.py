"""
PDF extraction module for the offline multimodal RAG system.

This module provides functionality to extract text from PDF files,
handling both native text and scanned image pages. It uses block-based
text extraction to reasonably handle multi-column layouts.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF


class PDFExtractor:
    """Extract text from PDF files with OCR detection for scanned pages."""

    def __init__(self, text_threshold: float = 0.1):
        """
        Initialize the PDF extractor.

        Args:
            text_threshold: Minimum ratio of extracted text to page area
                          to consider a page as having extractable text.
                          Pages below this threshold are flagged as needing OCR.
                          Default: 0.1 (10%)
        """
        self.text_threshold = text_threshold

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from all pages in a PDF file.

        Handles both text-based PDFs and image-based (scanned) PDFs.
        Multi-column layouts are processed using block-based extraction.

        Args:
            pdf_path: Path to the PDF file to extract from.

        Returns:
            List of dictionaries, each containing:
                - text: Extracted text from the page
                - page_number: Page number (1-indexed)
                - source: Original PDF filename
                - needs_ocr: Boolean indicating if page appears to be scanned
                           and needs OCR processing

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            fitz.FileError: If the file is not a valid PDF.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            document = fitz.open(pdf_path)
        except fitz.FileError as e:
            raise fitz.FileError(f"Invalid PDF file: {pdf_path}: {e}") from e

        results = []
        filename = pdf_path.name

        for page_num in range(len(document)):
            page = document[page_num]

            # Extract text using block-based method for multi-column support
            text = self._extract_text_from_page(page)

            # Determine if page needs OCR
            needs_ocr = self._should_flag_for_ocr(page, text)

            results.append(
                {
                    "text": text,
                    "page_number": page_num + 1,  # 1-indexed
                    "source": filename,
                    "needs_ocr": needs_ocr,
                }
            )

        document.close()
        return results

    def _extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a single page using block-based extraction.

        This method extracts text blocks in reading order, which helps
        preserve the layout and structure of multi-column documents.

        Args:
            page: The PDF page object to extract text from.

        Returns:
            Extracted text as a string.
        """
        text_blocks = []

        # Get text blocks sorted by y-coordinate and x-coordinate
        # This maintains a reasonable reading order for multi-column layouts
        blocks = page.get_text("blocks")

        for block in blocks:
            # block is a tuple: (x0, y0, x1, y1, text, block_num, block_type)
            # block_type: 0=text, 1=image
            if len(block) >= 5:
                text = block[4]  # Extract text from block

                # Clean up text: remove excessive whitespace
                if isinstance(text, str):
                    text = text.strip()
                    if text:
                        text_blocks.append(text)

        # Join blocks with newlines to maintain structure
        return "\n".join(text_blocks)

    def _should_flag_for_ocr(self, page: fitz.Page, extracted_text: str) -> bool:
        """
        Determine if a page should be flagged for OCR processing.

        A page is flagged if it appears to be a scanned image with
        little to no extractable text.

        Args:
            page: The PDF page object to check.
            extracted_text: The extracted text from the page.

        Returns:
            True if page should be OCR'd, False if sufficient text was extracted.
        """
        # Calculate page area
        page_rect = page.rect
        page_area = page_rect.get_area()

        if page_area == 0:
            return True

        # Estimate text area based on character count
        # Rough approximation: ~12 pixels per character on average
        text_area = len(extracted_text) * 12

        # Calculate ratio of text area to page area
        text_ratio = text_area / page_area

        # Flag for OCR if text ratio is below threshold
        if text_ratio < self.text_threshold:
            return True

        return False

def extract_pdf(pdf_path: str) -> List[Dict]:
    """
    Convenience function to extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries with extracted text and metadata.
    """
    extractor = PDFExtractor()
    return extractor.extract_text_from_pdf(pdf_path)


if __name__ == "__main__":
    """
    Standalone test of the PDF extractor.

    Usage:
        python pdf_extractor.py <path_to_pdf>
    """
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        print("\nExample:")
        print("  python pdf_extractor.py sample.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"Extracting text from: {pdf_path}\n")

    try:
        extractor = PDFExtractor()
        results = extractor.extract_text_from_pdf(pdf_path)

        print(f"Successfully extracted {len(results)} pages\n")
        print("=" * 80)

        for result in results:
            page_num = result["page_number"]
            needs_ocr = result["needs_ocr"]
            text = result["text"]
            source = result["source"]

            ocr_status = "[NEEDS OCR]" if needs_ocr else "[TEXT]"

            print(f"\nPage {page_num} {ocr_status} (Source: {source})")
            print("-" * 80)

            # Print first 500 characters of extracted text
            preview = text[:500] if text else "[NO TEXT EXTRACTED]"
            print(preview)

            if len(text) > 500:
                print(f"\n... ({len(text) - 500} more characters)")

        print("\n" + "=" * 80)
        print(f"\nExtraction Summary:")
        print(f"  Total pages: {len(results)}")

        ocr_pages = sum(1 for r in results if r["needs_ocr"])
        text_pages = len(results) - ocr_pages

        print(f"  Text pages: {text_pages}")
        print(f"  Pages needing OCR: {ocr_pages}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Invalid PDF file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

"""
Word document extraction module for the offline multimodal RAG system.

This module provides functionality to extract text from DOCX files,
including both paragraphs and tables.
"""

import sys
from pathlib import Path
from typing import List, Dict
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


class DOCXExtractor:
    """Extract text from Word documents (.docx files)."""

    def extract_text_from_docx(self, docx_path: str) -> List[Dict]:
        """
        Extract text from all paragraphs and tables in a DOCX file.

        Args:
            docx_path: Path to the DOCX file to extract from.

        Returns:
            List of dictionaries, each containing:
                - text: Extracted text from paragraph or table
                - paragraph_index: Index in the extraction sequence (0-indexed)
                - source: Original DOCX filename
                - content_type: Type of content ("paragraph" or "table")

        Raises:
            FileNotFoundError: If the DOCX file does not exist.
            Exception: If the file cannot be opened as a valid DOCX.
        """
        docx_path = Path(docx_path)

        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        try:
            document = Document(docx_path)
        except Exception as e:
            raise Exception(f"Invalid or corrupted DOCX file: {docx_path}") from e

        results = []
        filename = docx_path.name
        content_index = 0

        # Process paragraphs and tables in order
        for element in document.element.body:
            # Check if element is a paragraph
            if element.tag.endswith("p"):
                para = Paragraph(element, document)
                text = self._extract_paragraph_text(para)

                # Skip empty paragraphs
                if text.strip():
                    results.append(
                        {
                            "text": text,
                            "paragraph_index": content_index,
                            "source": filename,
                            "content_type": "paragraph",
                        }
                    )
                    content_index += 1

            # Check if element is a table
            elif element.tag.endswith("tbl"):
                table = Table(element, document)
                text = self._extract_table_text(table)

                # Skip empty tables
                if text.strip():
                    results.append(
                        {
                            "text": text,
                            "paragraph_index": content_index,
                            "source": filename,
                            "content_type": "table",
                        }
                    )
                    content_index += 1

        return results

    def _extract_paragraph_text(self, paragraph: Paragraph) -> str:
        """
        Extract text from a paragraph.

        Args:
            paragraph: The paragraph object to extract from.

        Returns:
            The extracted text from the paragraph.
        """
        return paragraph.text

    def _extract_table_text(self, table: Table) -> str:
        """
        Extract text from a table, preserving structure.

        Args:
            table: The table object to extract from.

        Returns:
            Text representation of the table with rows and cells separated.
        """
        rows_text = []

        for row in table.rows:
            cells_text = []
            for cell in row.cells:
                # Extract text from all paragraphs in the cell
                cell_text = "\n".join(p.text for p in cell.paragraphs)
                cells_text.append(cell_text.strip())

            # Join cells in a row with pipe separators for readability
            row_text = " | ".join(cells_text)
            rows_text.append(row_text)

        # Join rows with newlines
        return "\n".join(rows_text)

def extract_docx(docx_path: str) -> List[Dict]:
    """
    Convenience function to extract text from a DOCX file.

    Args:
        docx_path: Path to the DOCX file.

    Returns:
        List of dictionaries with extracted text and metadata.
    """
    extractor = DOCXExtractor()
    return extractor.extract_text_from_docx(docx_path)


if __name__ == "__main__":
    """
    Standalone test of the DOCX extractor.

    Usage:
        python docx_extractor.py <path_to_docx>
    """
    if len(sys.argv) < 2:
        print("Usage: python docx_extractor.py <path_to_docx>")
        print("\nExample:")
        print("  python docx_extractor.py sample.docx")
        sys.exit(1)

    docx_path = sys.argv[1]

    print(f"Extracting text from: {docx_path}\n")

    try:
        extractor = DOCXExtractor()
        results = extractor.extract_text_from_docx(docx_path)

        print(f"Successfully extracted {len(results)} content blocks\n")
        print("=" * 80)

        for result in results:
            index = result["paragraph_index"]
            content_type = result["content_type"]
            text = result["text"]
            source = result["source"]

            type_label = f"[{content_type.upper()}]"

            print(f"\nContent {index} {type_label} (Source: {source})")
            print("-" * 80)

            # Print first 500 characters of extracted text
            preview = text[:500] if text else "[NO TEXT EXTRACTED]"
            print(preview)

            if len(text) > 500:
                print(f"\n... ({len(text) - 500} more characters)")

        print("\n" + "=" * 80)
        print(f"\nExtraction Summary:")
        print(f"  Total content blocks: {len(results)}")

        paragraph_count = sum(1 for r in results if r["content_type"] == "paragraph")
        table_count = sum(1 for r in results if r["content_type"] == "table")

        print(f"  Paragraphs: {paragraph_count}")
        print(f"  Tables: {table_count}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

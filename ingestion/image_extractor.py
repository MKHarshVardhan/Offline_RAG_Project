"""
Image OCR extraction module for the offline multimodal RAG system.

Extracts text from PNG/JPG/JPEG images using pytesseract with basic
pre-processing to improve OCR accuracy on scanned/low-quality images.
"""

import sys
from pathlib import Path
from typing import Dict

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def extract_image(image_path: str) -> Dict:
    """
    Extract text from an image file using OCR.

    Applies grayscale conversion, contrast enhancement, and thresholding
    before running Tesseract OCR.

    Args:
        image_path: Path to the image file (png/jpg/jpeg).

    Returns:
        Dict with keys:
            - text: Extracted text string
            - confidence: Mean OCR confidence score (0.0–100.0)
            - source: Image filename

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    img = Image.open(path)

    # Pre-processing: grayscale → contrast boost → threshold
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.point(lambda p: 255 if p > 128 else 0)

    # Run OCR with verbose data to extract confidence scores
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Filter valid confidence values (pytesseract returns -1 for non-word entries)
    confidences = [int(c) for c in data["conf"] if int(c) != -1]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0

    text = pytesseract.image_to_string(img).strip()

    return {
        "text": text,
        "confidence": round(confidence, 2),
        "source": path.name,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_extractor.py <path_to_image>")
        sys.exit(1)

    result = extract_image(sys.argv[1])
    print(f"Source    : {result['source']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Text      :\n{result['text'] or '[NO TEXT EXTRACTED]'}")

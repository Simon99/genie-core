from __future__ import annotations

import subprocess
from pathlib import Path


def split_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    fmt: str = "png",
) -> list[dict]:
    """Split each page of a PDF into individual image files.

    Uses sips (macOS built-in) or falls back to python if available.
    Returns list of {"page": int, "path": str}.
    """
    pdf_path = str(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _split_with_pymupdf(pdf_path, output_dir, dpi, fmt)
    except ImportError:
        return _split_with_sips(pdf_path, output_dir, fmt)


def _split_with_pymupdf(pdf_path: str, output_dir: Path, dpi: int, fmt: str) -> list[dict]:
    import fitz

    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        out_file = output_dir / f"page_{page_num + 1:04d}.{fmt}"
        pix.save(str(out_file))
        results.append({"page": page_num + 1, "path": str(out_file)})

    doc.close()
    return results


def _split_with_sips(pdf_path: str, output_dir: Path, fmt: str) -> list[dict]:
    """Fallback: use macOS sips to convert PDF pages."""
    cmd = [
        "sips", "-s", "format", fmt,
        pdf_path, "--out", str(output_dir)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    results = []
    for i, f in enumerate(sorted(output_dir.glob(f"*.{fmt}")), 1):
        results.append({"page": i, "path": str(f)})
    return results

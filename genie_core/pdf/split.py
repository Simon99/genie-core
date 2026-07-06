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

    Uses PyMuPDF as the primary path; falls back to macOS sips only for
    single-page PDFs (sips silently rasterizes only the first page of a
    multi-page PDF).
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
    try:
        results = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            out_file = output_dir / f"page_{page_num + 1:04d}.{fmt}"
            pix.save(str(out_file))
            results.append({"page": page_num + 1, "path": str(out_file)})

        return results
    finally:
        doc.close()


def _sips_page_count(pdf_path: str) -> int | None:
    """Return the page count via `sips -g pdfNPages`, or None if unavailable."""
    try:
        result = subprocess.run(
            ["sips", "-g", "pdfNPages", pdf_path],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "pdfNPages:" in line:
            try:
                return int(line.split("pdfNPages:")[1].strip())
            except ValueError:
                return None
    return None


def _split_with_sips(pdf_path: str, output_dir: Path, fmt: str) -> list[dict]:
    """Fallback: use macOS sips to convert a single-page PDF.

    sips only rasterizes the first page of a multi-page PDF, so this path
    refuses multi-page input instead of silently returning one page.
    """
    n_pages = _sips_page_count(pdf_path)
    if n_pages is None or n_pages > 1:
        raise RuntimeError(
            "PyMuPDF is required to split this PDF (%s): the sips fallback "
            "only handles single-page PDFs (detected pages: %s). "
            "Install it with: pip install PyMuPDF"
            % (pdf_path, n_pages if n_pages is not None else "unknown")
        )

    # Use a unique output prefix so we never pick up pre-existing files
    # in output_dir when globbing for the result.
    out_file = output_dir / f"page_0001.{fmt}"
    if out_file.exists():
        out_file.unlink()

    cmd = [
        "sips", "-s", "format", fmt,
        pdf_path, "--out", str(out_file)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    except FileNotFoundError:
        raise RuntimeError(
            "Neither PyMuPDF nor sips is available to split %s. "
            "Install PyMuPDF with: pip install PyMuPDF" % pdf_path
        )

    if not out_file.exists():
        raise RuntimeError("sips did not produce expected output: %s" % out_file)

    return [{"page": 1, "path": str(out_file)}]

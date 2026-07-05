"""Generate synthetic test fixtures for genie project.

Creates:
- A short test video (10s) with scene changes and synthetic audio
- A simple multi-page PDF with mixed content
"""
import subprocess
import sys
from pathlib import Path


def generate_test_video(output_path: str, duration: int = 10):
    """Generate a 10-second test video with 3 color scenes + sine wave audio."""
    scene_duration = duration // 3

    # 3 colored scenes with text overlay + sine wave audio
    cmd = [
        "ffmpeg",
        "-f", "lavfi", "-i", (
            f"color=c=red:s=1280x720:d={scene_duration}[v0];"
            f"color=c=blue:s=1280x720:d={scene_duration}[v1];"
            f"color=c=green:s=1280x720:d={scene_duration}[v2];"
            f"[v0][v1][v2]concat=n=3:v=1:a=0[vout];"
            f"[vout]drawtext=text='Scene %{{n}}':fontsize=48:fontcolor=white:x=(w-tw)/2:y=(h-th)/2[final]"
        ),
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-map", "[final]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        output_path, "-y"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Simpler fallback
        cmd = [
            "ffmpeg",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=1280x720:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            output_path, "-y"
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    print(f"Generated test video: {output_path}")


def generate_test_pdf(output_path: str):
    """Generate a 3-page test PDF with different content types."""
    try:
        import fitz
        doc = fitz.open()

        # Page 1: Title page
        page = doc.new_page(width=612, height=792)
        page.insert_text((200, 300), "Genie Test Document", fontsize=24)
        page.insert_text((200, 350), "Page 1 - Title", fontsize=14)

        # Page 2: Text content
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Chapter 1: Introduction", fontsize=18)
        page.insert_text((72, 110), "This is a test paragraph with some content.", fontsize=12)
        page.insert_text((72, 130), "It demonstrates multi-line text extraction.", fontsize=12)
        page.insert_text((72, 170), "Key Points:", fontsize=14)
        page.insert_text((90, 195), "• First item in the list", fontsize=12)
        page.insert_text((90, 215), "• Second item in the list", fontsize=12)
        page.insert_text((90, 235), "• Third item in the list", fontsize=12)

        # Page 3: Table-like content
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Data Summary", fontsize=18)
        page.insert_text((72, 110), "Name        | Value  | Status", fontsize=12)
        page.insert_text((72, 130), "------------|--------|--------", fontsize=12)
        page.insert_text((72, 150), "Alpha       | 100    | Active", fontsize=12)
        page.insert_text((72, 170), "Beta        | 200    | Pending", fontsize=12)
        page.insert_text((72, 190), "Gamma       | 300    | Done", fontsize=12)

        doc.save(output_path)
        doc.close()
        print(f"Generated test PDF: {output_path}")

    except ImportError:
        # Fallback: generate minimal PDF manually
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
trailer<</Size 5/Root 1 0 R>>
startxref
331
%%EOF"""
        Path(output_path).write_bytes(pdf_content)
        print(f"Generated minimal test PDF: {output_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_test_video(str(out_dir / "test_video.mp4"))
    generate_test_pdf(str(out_dir / "test_document.pdf"))
    print("All fixtures generated.")

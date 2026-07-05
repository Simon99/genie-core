"""Generate realistic test fixtures for genie project.

Creates:
- A test video (~15s) with scene changes + TTS speech (macOS `say`)
- A multi-page PDF with images, a flowchart, and titles
"""
import subprocess
import sys
import struct
import zlib
from pathlib import Path


def generate_test_video(output_path: str):
    """Generate a ~15s test video with 3 scenes, text overlays, and TTS narration."""
    out = Path(output_path)
    tmp = out.parent / "_video_tmp"
    tmp.mkdir(exist_ok=True)

    # Generate TTS audio segments with macOS say
    speeches = [
        ("Welcome to the Genie project demo. This is scene one.", "en", "scene1.aiff"),
        ("现在切换到第二个场景，我们来看一下数据分析的结果。", "zh-TW", "scene2.aiff"),
        ("Finally, let's review the summary and next steps.", "en", "scene3.aiff"),
    ]

    audio_files = []
    for text, voice_lang, fname in speeches:
        aiff = str(tmp / fname)
        wav = str(tmp / fname.replace(".aiff", ".wav"))
        voice = "Samantha" if voice_lang == "en" else "Mei-Jia"
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run([
            "ffmpeg", "-i", aiff, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav, "-y"
        ], capture_output=True, check=True)
        audio_files.append(wav)

    # Get duration of each audio segment
    durations = []
    for wav in audio_files:
        result = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav
        ]).decode().strip()
        durations.append(float(result))

    # Generate video segments with colored backgrounds and text
    colors = ["0x2C3E50", "0x2980B9", "0x27AE60"]
    titles = ["Scene 1 - Introduction", "Scene 2 - Data Analysis", "Scene 3 - Summary"]
    segment_files = []

    for i, (color, title, dur, wav) in enumerate(zip(colors, titles, durations, audio_files)):
        seg_file = str(tmp / f"seg{i}.mp4")
        text_file = str(tmp / f"text{i}.txt")
        with open(text_file, "w") as f:
            f.write(title)
        subprocess.run([
            "ffmpeg",
            "-f", "lavfi", "-i",
            f"color=c={color}:s=1280x720:d={dur}",
            "-i", wav,
            "-vf", f"drawtext=textfile={text_file}:fontsize=40:fontcolor=white:x=(w-tw)/2:y=(h-th)/2",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-shortest", seg_file, "-y"
        ], capture_output=True, check=True)
        segment_files.append(seg_file)

    # Concat segments
    concat_file = str(tmp / "concat.txt")
    with open(concat_file, "w") as f:
        for seg in segment_files:
            abs_seg = str(Path(seg).resolve())
            f.write(f"file '{abs_seg}'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy", output_path, "-y"
    ], capture_output=True, check=True)

    # Cleanup
    import shutil
    shutil.rmtree(tmp)

    total_dur = sum(durations)
    print(f"Generated test video: {output_path} ({total_dur:.1f}s, 3 scenes with TTS)")


def _create_png_bytes(width, height, r, g, b):
    """Create a minimal solid-color PNG in memory."""
    def make_chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = make_chunk(b"IDR" if False else b"IHDR",
                      struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    raw = b""
    row = bytes([r, g, b] * width)
    for _ in range(height):
        raw += b"\x00" + row

    compressed = zlib.compress(raw)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def _render_flowchart(output_path: str) -> str:
    """Render a state/flow diagram using Graphviz dot."""
    dot_source = """
digraph pipeline {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.3,0.15"];
    edge [fontname="Helvetica", fontsize=10];

    // States
    idle      [label="Idle\\n(Waiting for Input)", fillcolor="#E8F4FD", color="#2980B9", penwidth=2];
    ingest    [label="Ingesting\\n(Reading Files)", fillcolor="#FEF9E7", color="#F39C12", penwidth=2];

    // Decision
    node [shape=diamond];
    check     [label="Has\\nAudio?", fillcolor="#FADBD8", color="#E74C3C", penwidth=2];

    // Processing states
    node [shape=box];
    extract_a [label="Extract Audio\\n(ffmpeg)", fillcolor="#D5F5E3", color="#27AE60", penwidth=2];
    stt       [label="Speech-to-Text\\n(Whisper)", fillcolor="#D5F5E3", color="#27AE60", penwidth=2];
    extract_f [label="Extract Frames\\n(Scene Detection)", fillcolor="#D5F5E3", color="#27AE60", penwidth=2];
    ocr       [label="OCR / Vision Parse\\n(Qwen3-VL)", fillcolor="#D5F5E3", color="#27AE60", penwidth=2];
    merge     [label="Merge & Align\\n(Timestamps)", fillcolor="#EBF5FB", color="#3498DB", penwidth=2];
    llm       [label="LLM Summarize\\n(Qwen3.6-35B)", fillcolor="#F5EEF8", color="#8E44AD", penwidth=2];

    // Output states
    node [shape=box, style="rounded,filled,bold"];
    output_md [label="Markdown\\nOutput", fillcolor="#D4EFDF", color="#1E8449", penwidth=2];
    output_pdf[label="PDF\\nOutput", fillcolor="#D4EFDF", color="#1E8449", penwidth=2];

    // Error
    node [shape=octagon, style="filled"];
    error     [label="Error\\n(Retry/Skip)", fillcolor="#FDEDEC", color="#C0392B", penwidth=2];

    // Transitions
    idle      -> ingest    [label="file received"];
    ingest    -> check     [label="parsed"];
    check     -> extract_a [label="yes"];
    check     -> extract_f [label="no (PDF only)"];
    extract_a -> stt       [label="wav ready"];
    stt       -> merge     [label="transcript ready"];
    extract_f -> ocr       [label="frames ready"];
    ocr       -> merge     [label="page text ready"];
    merge     -> llm       [label="aligned data"];
    llm       -> output_md [label="structured"];
    llm       -> output_pdf[label="structured"];
    output_md -> idle      [label="done", style=dashed];
    output_pdf-> idle      [label="done", style=dashed];

    // Error edges
    extract_a -> error     [label="fail", style=dotted, color="#E74C3C"];
    stt       -> error     [label="fail", style=dotted, color="#E74C3C"];
    error     -> idle      [label="reset", style=dashed, color="#E74C3C"];
}
"""
    dot_file = output_path.replace(".png", ".dot")
    with open(dot_file, "w") as f:
        f.write(dot_source)

    try:
        subprocess.run(
            ["dot", "-Tpng", "-Gdpi=150", dot_file, "-o", output_path],
            check=True, capture_output=True
        )
        Path(dot_file).unlink(missing_ok=True)
        return output_path
    except Exception:
        Path(dot_file).unlink(missing_ok=True)
        return None


def generate_test_pdf(output_path: str):
    """Generate a 3-page PDF with images, flowchart, and titles."""
    import fitz

    doc = fitz.open()

    # --- Page 1: Title + 3 images ---
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Genie Project - Test Document", fontsize=24,
                     color=(0.1, 0.2, 0.4))
    page.insert_text((72, 90), "Multimodal Content Extraction Demo", fontsize=14,
                     color=(0.4, 0.4, 0.4))

    # Insert 3 colored rectangles as "images"
    colors = [(0.9, 0.3, 0.3), (0.3, 0.7, 0.3), (0.3, 0.3, 0.9)]
    labels = ["Chart A: Revenue", "Chart B: Users", "Chart C: Growth"]
    x_positions = [72, 232, 392]

    for x, color, label in zip(x_positions, colors, labels):
        rect = fitz.Rect(x, 120, x + 140, 280)
        page.draw_rect(rect, color=color, fill=color, width=0)
        # Add axis-like lines inside
        page.draw_line((x + 10, 270), (x + 130, 270), color=(1, 1, 1), width=1)
        page.draw_line((x + 10, 140), (x + 10, 270), color=(1, 1, 1), width=1)
        # Fake bar chart lines
        for bx in range(30, 121, 30):
            bar_h = (bx * 2) % 100 + 20
            page.draw_rect(fitz.Rect(x + bx - 8, 270 - bar_h, x + bx + 8, 270),
                          color=(1, 1, 1), fill=(1, 1, 1, 0.7), width=0)
        page.insert_text((x + 10, 300), label, fontsize=10, color=(0.2, 0.2, 0.2))

    page.insert_text((72, 340), "Figure 1: Q2 Performance Metrics", fontsize=11,
                     color=(0.3, 0.3, 0.3))
    page.insert_text((72, 380), "Key Findings:", fontsize=14, color=(0.1, 0.1, 0.1))
    page.insert_text((90, 405), "• Revenue increased by 23% compared to Q1", fontsize=11)
    page.insert_text((90, 425), "• Monthly active users reached 1.2M milestone", fontsize=11)
    page.insert_text((90, 445), "• User growth rate accelerated in APAC region", fontsize=11)

    # --- Page 2: State diagram rendered by Graphviz ---
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "System Architecture - Data Pipeline", fontsize=20,
                     color=(0.1, 0.2, 0.4))

    flowchart_png = _render_flowchart(str(Path(output_path).parent / "_flowchart.png"))
    if flowchart_png:
        img_rect = fitz.Rect(36, 80, 576, 750)
        page.insert_image(img_rect, filename=flowchart_png)
        Path(flowchart_png).unlink(missing_ok=True)

    # --- Page 3: Data table + notes ---
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Test Results Summary", fontsize=20,
                     color=(0.1, 0.2, 0.4))

    # Table header
    page.insert_text((72, 110), "Model", fontsize=12, color=(0.1, 0.1, 0.1))
    page.insert_text((220, 110), "Accuracy", fontsize=12, color=(0.1, 0.1, 0.1))
    page.insert_text((340, 110), "Speed (s)", fontsize=12, color=(0.1, 0.1, 0.1))
    page.insert_text((460, 110), "Status", fontsize=12, color=(0.1, 0.1, 0.1))
    page.draw_line((72, 118), (540, 118), color=(0.3, 0.3, 0.3), width=1)

    rows = [
        ("Whisper Medium", "94.2%", "12.3", "Passed"),
        ("Whisper Large", "96.8%", "28.7", "Passed"),
        ("Qwen3-VL 30B", "91.5%", "8.4", "Passed"),
        ("Qwen3.6 35B-A3B", "89.1%", "5.2", "Review"),
    ]
    for i, (model, acc, speed, status) in enumerate(rows):
        y = 140 + i * 25
        page.insert_text((72, y), model, fontsize=11)
        page.insert_text((220, y), acc, fontsize=11)
        page.insert_text((340, y), speed, fontsize=11)
        status_color = (0.2, 0.6, 0.2) if status == "Passed" else (0.8, 0.5, 0.1)
        page.insert_text((460, y), status, fontsize=11, color=status_color)

    page.draw_line((72, 245), (540, 245), color=(0.7, 0.7, 0.7), width=0.5)

    page.insert_text((72, 280), "Notes:", fontsize=14, color=(0.1, 0.1, 0.1))
    page.insert_text((72, 305), "1. All tests conducted on Apple M3 with 18GB unified memory.",
                     fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((72, 325), "2. Accuracy measured on mixed Chinese-English meeting corpus.",
                     fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((72, 345), "3. Speed is wall-clock time for a 5-minute recording.",
                     fontsize=10, color=(0.3, 0.3, 0.3))

    doc.save(output_path)
    doc.close()
    print(f"Generated test PDF: {output_path} (3 pages: charts+images, flowchart, data table)")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_test_video(str(out_dir / "test_video.mp4"))
    generate_test_pdf(str(out_dir / "test_document.pdf"))
    print("All fixtures generated.")

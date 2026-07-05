"""Generate an advanced test video that matches the test PDF content.

Each PDF page gets a corresponding narration scene, so we can verify
that the meeting analyzer correctly maps speech to slide content.

Page 1: Charts + Key Findings → narration describes revenue and user metrics
Page 2: Flowchart → narration explains the data pipeline architecture
Page 3: Data Table → narration discusses test results and model comparisons
"""
import subprocess
import sys
from pathlib import Path


NARRATIONS = [
    {
        "voice": "Samantha",
        "text": (
            "Let's start with our Q2 performance overview. "
            "As you can see on this slide, revenue increased by 23 percent compared to Q1. "
            "Our monthly active users have reached the 1.2 million milestone, "
            "and the user growth rate has accelerated significantly in the APAC region. "
            "Chart A shows revenue trending upward. Chart B shows user count. "
            "And Chart C confirms the growth trajectory we predicted last quarter."
        ),
        "color": "0x2C3E50",
        "title": "Page 1 - Q2 Performance Review",
    },
    {
        "voice": "Samantha",
        "text": (
            "Now let me walk you through our system architecture. "
            "The data pipeline starts with the video input, which gets split into two parallel tracks. "
            "On the left side, we extract audio and run it through Whisper for speech to text. "
            "On the right side, we extract frames and use scene detection to find key moments. "
            "There's a decision diamond that checks whether the input has audio. "
            "If it's a PDF only input with no audio, we skip the speech track entirely. "
            "Both tracks merge at the alignment stage, then the LLM summarizer produces the final output. "
            "Error handling is built in. If extraction fails, we retry or skip that segment."
        ),
        "color": "0x1A5276",
        "title": "Page 2 - System Architecture",
    },
    {
        "voice": "Samantha",
        "text": (
            "Finally, let's review our model benchmark results. "
            "Whisper Medium achieved 94.2 percent accuracy at 12.3 seconds processing time. "
            "Whisper Large pushed accuracy to 96.8 percent but took 28.7 seconds, more than double. "
            "For vision tasks, Qwen3 VL 30B scored 91.5 percent accuracy at just 8.4 seconds. "
            "The Qwen 3.6 35B A3B model for text tasks had 89.1 percent accuracy at 5.2 seconds, "
            "but it's still under review because we noticed some edge cases with mixed language input. "
            "All tests were conducted on an Apple M3 with 18 gigabytes of unified memory. "
            "The accuracy was measured on our mixed Chinese English meeting corpus."
        ),
        "color": "0x1E8449",
        "title": "Page 3 - Test Results",
    },
]


def generate_advanced_video(output_path: str):
    out = Path(output_path)
    tmp = out.parent / "_adv_video_tmp"
    tmp.mkdir(exist_ok=True)

    segment_files = []

    for i, narr in enumerate(NARRATIONS):
        print("  Generating scene %d: %s" % (i + 1, narr["title"]))

        # Generate TTS
        aiff = str(tmp / ("narr_%d.aiff" % i))
        wav = str(tmp / ("narr_%d.wav" % i))
        subprocess.run(["say", "-v", narr["voice"], "-o", aiff, narr["text"]], check=True)
        subprocess.run([
            "ffmpeg", "-i", aiff, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav, "-y"
        ], capture_output=True, check=True)

        # Get audio duration
        dur = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav
        ]).decode().strip()
        duration = float(dur)
        print("    Duration: %.1fs" % duration)

        # Generate video segment with title overlay
        seg_file = str(tmp / ("seg_%d.mp4" % i))
        text_file = str(tmp / ("text_%d.txt" % i))
        with open(text_file, "w") as f:
            f.write(narr["title"])

        subprocess.run([
            "ffmpeg",
            "-f", "lavfi", "-i",
            "color=c=%s:s=1280x720:d=%s" % (narr["color"], duration),
            "-i", wav,
            "-vf",
            "drawtext=textfile=%s:fontsize=36:fontcolor=white:x=(w-tw)/2:y=(h-th)/2" % text_file,
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-shortest", seg_file, "-y"
        ], capture_output=True, check=True)

        segment_files.append(seg_file)

    # Concat all segments
    concat_file = str(tmp / "concat.txt")
    with open(concat_file, "w") as f:
        for seg in segment_files:
            f.write("file '%s'\n" % str(Path(seg).resolve()))

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy", output_path, "-y"
    ], capture_output=True, check=True)

    # Get total duration
    total = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", output_path
    ]).decode().strip()

    # Cleanup
    import shutil
    shutil.rmtree(tmp)

    print("Generated advanced test video: %s (%.1fs, %d scenes)" % (output_path, float(total), len(NARRATIONS)))


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_advanced_video(str(out_dir / "test_video_advanced.mp4"))

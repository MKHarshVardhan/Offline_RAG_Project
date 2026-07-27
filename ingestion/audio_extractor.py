"""
Audio transcription module for the offline multimodal RAG system.

Transcribes audio files using faster-whisper with a locally cached model.
Model files are downloaded on first use and cached; no internet required at runtime.

Note: Only .wav is natively supported by faster-whisper without extra dependencies.
      For .mp3 / .m4a files, convert to .wav first using ffmpeg:
          ffmpeg -i input.mp3 output.wav
"""

import sys
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel

# Model is downloaded once and cached in ~/.cache/huggingface/hub by default.
# Change cache_dir to a local folder (e.g. "./models/whisper") to keep it
# inside the project and guarantee fully offline use after first download.
_MODEL_SIZE = "base"
_CACHE_DIR = "./models/whisper"


def _load_model() -> WhisperModel:
    """Load the Whisper model from local cache (CPU, int8 for low memory use)."""
    return WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8", download_root=_CACHE_DIR)


def extract_audio(audio_path: str) -> Dict:
    """
    Transcribe an audio file and return text with per-segment timestamps.

    Args:
        audio_path: Path to a .wav or .mp3 audio file.
                    .mp3 files require ffmpeg installed on the system PATH.

    Returns:
        Dict with keys:
            - text:     Full concatenated transcript string
            - segments: List of {"start": float, "end": float, "text": str}
            - source:   Audio filename

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError:        If the file extension is not .wav or .mp3.
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    model = _load_model()

    # beam_size=5 is the faster-whisper default; vad_filter removes silence
    segments_iter, _ = model.transcribe(str(path), beam_size=5, vad_filter=True)

    segments: List[Dict] = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })

    full_text = " ".join(s["text"] for s in segments)

    return {
        "text": full_text,
        "segments": segments,
        "source": path.name,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_extractor.py <path_to_audio>")
        sys.exit(1)

    result = extract_audio(sys.argv[1])
    print(f"Source : {result['source']}")
    print(f"Text   : {result['text'] or '[NO TRANSCRIPT]'}")
    print(f"\nSegments ({len(result['segments'])}):")
    for seg in result["segments"]:
        print(f"  [{seg['start']:6.2f}s -> {seg['end']:6.2f}s]  {seg['text']}")

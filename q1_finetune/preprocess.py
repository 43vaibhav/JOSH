"""
Q1a — Preprocessing
====================
Downloads transcription JSONs from GCS, cleans & filters segments,
and writes a train_manifest.json used by train.py and the Colab notebook.

Usage:
    python preprocess.py

Output:
    q1_finetune/processed/train_manifest.json
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_PATH    = ROOT / "data" / "FT_Data.xlsx"
OUTPUT_DIR   = Path(__file__).parent / "processed"
MANIFEST_OUT = OUTPUT_DIR / "train_manifest.json"

# ── URL rewriting ─────────────────────────────────────────────────
# Spreadsheet URLs point to a private bucket; accessible bucket uses upload_goai
_OLD = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi"
_NEW = "https://storage.googleapis.com/upload_goai"

def fix_url(url: str) -> str:
    """Rewrite private GCS URL → public upload_goai URL."""
    if not isinstance(url, str):
        return ""
    m = re.search(r"/hi/(\d+)/(\d+_[^/]+)$", url)
    if not m:
        return url
    return f"{_NEW}/{m.group(1)}/{m.group(2)}"


# ── Text cleaning ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single transcript segment:
      1. NFC-normalise Unicode
      2. Strip zero-width characters (ZWNJ / ZWJ / BOM) common in Devanagari
      3. Collapse repeated dandas (।।→।)
      4. Remove leading/trailing filler hyphens  (-- at start/end)
      5. Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[।]{2,}", "।", text)
    text = re.sub(r"^\s*-+\s*", "", text)
    text = re.sub(r"\s*-+\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Segment filtering ─────────────────────────────────────────────

# Whisper context window = 30 s; skip anything too short to be useful
MIN_DURATION = 0.5   # seconds
MAX_DURATION = 29.0  # seconds
MIN_CHARS    = 3     # skip single backchannels (हाँ, जी, हूँ)

def filter_segments(segments: list[dict]) -> list[dict]:
    """Keep only segments that are usable for ASR training."""
    good = []
    for seg in segments:
        dur  = seg.get("end", 0) - seg.get("start", 0)
        text = clean_text(seg.get("text", ""))
        if dur < MIN_DURATION or dur > MAX_DURATION:
            continue
        if len(text) < MIN_CHARS:
            continue
        seg["text"]     = text
        seg["duration"] = round(dur, 3)
        good.append(seg)
    return good


# ── Per-recording processing ──────────────────────────────────────

def process_recording(row: pd.Series) -> list[dict]:
    """Fetch transcription JSON, clean, filter, return manifest entries."""
    url = fix_url(str(row["transcription_url_gcp"]))
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        segments = resp.json()
    except Exception as e:
        log.warning(f"  SKIP {row['recording_id']} — {e}")
        return []

    entries = []
    for seg in filter_segments(segments):
        entries.append({
            "recording_id": str(row["recording_id"]),
            "audio_url":    fix_url(str(row["rec_url_gcp"])),
            "start":        seg["start"],
            "end":          seg["end"],
            "text":         seg["text"],
            "duration":     seg["duration"],
            "language":     str(row.get("language", "hi")),
        })
    return entries


# ── Main ──────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load spreadsheet
    df = pd.read_excel(DATA_PATH)
    df.columns = [c.strip() for c in df.columns if isinstance(c, str)]
    df = df.dropna(subset=["recording_id", "rec_url_gcp"])
    df["recording_id"] = df["recording_id"].astype(int).astype(str)
    log.info(f"Loaded {len(df)} recordings")

    # Show URL rewrite example
    sample = df.iloc[0]
    print("\n── URL Rewrite Example ──────────────────────────────────")
    print(f"  Before : {sample['transcription_url_gcp']}")
    print(f"  After  : {fix_url(str(sample['transcription_url_gcp']))}")

    # Concurrent download of all transcription JSONs
    print(f"\n── Downloading & Processing {len(df)} transcriptions ────")
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_recording, row): row["recording_id"]
                   for _, row in df.iterrows()}
        done = 0
        for fut in as_completed(futures):
            manifest.extend(fut.result())
            done += 1
            if done % 10 == 0 or done == len(df):
                print(f"  {done}/{len(df)} recordings done  ({len(manifest)} segments so far)")

    # Save manifest
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Summary
    durations = [m["duration"] for m in manifest]
    print("\n── Preprocessing Summary ───────────────────────────────")
    print(f"  Recordings processed : {len(df)}")
    print(f"  Segments kept        : {len(manifest):,}")
    print(f"  Total audio          : {sum(durations)/3600:.2f} hours")
    print(f"  Avg segment duration : {np.mean(durations):.2f}s")
    print(f"  Median               : {np.median(durations):.2f}s")
    print(f"  Manifest saved to    : {MANIFEST_OUT}")


if __name__ == "__main__":
    main()

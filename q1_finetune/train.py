"""
Q1b — Fine-tune Whisper-small on Hindi ASR
===========================================
Runs AFTER preprocess.py has produced processed/train_manifest.json.

Features:
  • Auto-detects transformers version (eval_strategy vs evaluation_strategy)
  • Uses jiwer for WER — fully offline, no Hub download
  • Disk-caches the processed dataset so re-runs are instant
  • Progress bar during first-time GCS audio download

Usage:
    python train.py

Output:
    q1_finetune/whisper-small-hindi-ft/   ← fine-tuned model checkpoint
"""

import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict
from jiwer import wer as jiwer_wer
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── Paths ─────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
MANIFEST_PATH = Path(__file__).parent / "processed" / "train_manifest.json"
CACHE_DIR     = Path(__file__).parent / "processed" / "dataset_cache"
OUTPUT_DIR    = Path(__file__).parent / "whisper-small-hindi-ft"

# ── Model config ──────────────────────────────────────────────────
MODEL_ID = "openai/whisper-small"
LANGUAGE = "Hindi"
TASK     = "transcribe"

# ── Training arguments ────────────────────────────────────────────
# transformers >= 4.41 renamed evaluation_strategy → eval_strategy
_NEW_API = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 41)

_kwargs = dict(
    output_dir                = str(OUTPUT_DIR),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,          # effective batch = 16
    learning_rate             = 1e-5,
    warmup_steps              = 200,
    max_steps                 = 2000,
    gradient_checkpointing    = True,
    fp16                      = torch.cuda.is_available(),
    per_device_eval_batch_size= 8,
    predict_with_generate     = True,
    generation_max_length     = 225,
    save_steps                = 500,
    eval_steps                = 500,
    logging_steps             = 50,
    report_to                 = ["tensorboard"],
    load_best_model_at_end    = True,
    metric_for_best_model     = "wer",
    greater_is_better         = False,
    push_to_hub               = False,
)
_kwargs["eval_strategy" if _NEW_API else "evaluation_strategy"] = "steps"
TRAINING_ARGS = Seq2SeqTrainingArguments(**_kwargs)


# ── Data collator ─────────────────────────────────────────────────

@dataclass
class DataCollator:
    """Pads audio features and label token IDs to uniform length within a batch."""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad mel spectrograms
        audio_batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        # Pad label sequences; use -100 so loss ignores padding positions
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper prepends a BOS token — strip it; the model adds it internally
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        audio_batch["labels"] = labels
        return audio_batch


# ── WER metric (offline via jiwer) ───────────────────────────────

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalise whitespace before scoring
    pred_str  = [" ".join(s.split()) for s in pred_str]
    label_str = [" ".join(s.split()) for s in label_str]

    return {"wer": round(jiwer_wer(label_str, pred_str), 4)}


# ── Dataset builder ───────────────────────────────────────────────

def load_or_build_dataset(manifest_path: Path, proc, cache_dir: Path) -> DatasetDict:
    """
    First run  : downloads audio segments from GCS, extracts features,
                 saves processed dataset to disk  (takes ~30-45 min).
    Later runs : loads instantly from disk cache  (takes ~5 sec).
    """
    train_cache = cache_dir / "train"
    val_cache   = cache_dir / "validation"

    if train_cache.exists() and val_cache.exists():
        print("  ✓ Disk cache found — loading instantly (no GCS download needed)")
        return DatasetDict({
            "train":      Dataset.load_from_disk(str(train_cache)),
            "validation": Dataset.load_from_disk(str(val_cache)),
        })

    print("  No cache found — downloading audio from GCS (one-time only)")
    import requests, soundfile as sf

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # 90 / 10 stratified split by recording_id
    rec_ids = list({m["recording_id"] for m in manifest})
    np.random.seed(42)
    np.random.shuffle(rec_ids)
    train_ids = set(rec_ids[: int(len(rec_ids) * 0.9)])

    train_entries = [e for e in manifest if     e["recording_id"] in train_ids]
    val_entries   = [e for e in manifest if not e["recording_id"] in train_ids]

    def featurise(entries: list, split: str) -> list:
        rows, skipped = [], 0
        for entry in tqdm(entries, desc=f"  {split}", unit="seg"):
            try:
                resp = requests.get(entry["audio_url"], timeout=60)
                resp.raise_for_status()
                data, sr = sf.read(io.BytesIO(resp.content))

                # Slice segment
                seg = data[int(entry["start"] * sr) : int(entry["end"] * sr)]
                if seg.ndim > 1:
                    seg = seg.mean(axis=1)   # stereo → mono

                # Resample to 16 kHz if needed
                if sr != 16000:
                    import librosa
                    seg = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr  = 16000

                feats  = proc.feature_extractor(
                    seg.astype(np.float32), sampling_rate=sr, return_tensors="pt"
                ).input_features[0]
                labels = proc.tokenizer(entry["text"]).input_ids
                rows.append({"input_features": feats, "labels": labels})
            except Exception:
                skipped += 1
        print(f"  {split}: {len(rows):,} ok  |  {skipped} skipped")
        return rows

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = Dataset.from_list(featurise(train_entries, "Train"))
    val_ds   = Dataset.from_list(featurise(val_entries,   "Val  "))

    print("  Saving dataset to disk cache …")
    train_ds.save_to_disk(str(train_cache))
    val_ds.save_to_disk(str(val_cache))
    print(f"  ✓ Cache saved → {cache_dir}")

    return DatasetDict({"train": train_ds, "validation": val_ds})


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("─" * 60)
    print(f"  transformers : {transformers.__version__}")
    print(f"  torch        : {torch.__version__}")
    print(f"  CUDA         : {torch.cuda.is_available()}")
    print("─" * 60)

    # 1. Load processor + model
    print("\n[1/4] Loading Whisper-small …")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []
    model.config.use_cache        = False   # required for gradient checkpointing

    # 2. Build / load dataset
    print("\n[2/4] Loading dataset …")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run  python preprocess.py  first."
        )
    dataset = load_or_build_dataset(MANIFEST_PATH, processor, CACHE_DIR)
    print(f"  Train : {len(dataset['train']):,}  |  Val : {len(dataset['validation']):,}")

    # 3. Trainer
    print("\n[3/4] Setting up trainer …")
    collator = DataCollator(processor=processor)
    trainer  = Seq2SeqTrainer(
        args            = TRAINING_ARGS,
        model           = model,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["validation"],
        data_collator   = collator,
        compute_metrics = compute_metrics,
        tokenizer       = processor.feature_extractor,
    )

    # 4. Train
    print("\n[4/4] Training …")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"\n✓ Done — model saved to {OUTPUT_DIR}")

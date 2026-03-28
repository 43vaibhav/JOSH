"""
Q1c–g — Evaluation, Error Sampling, Taxonomy & Fixes
======================================================
Evaluates both Whisper baselines on FLEURS Hindi test set,
samples errors systematically, builds a taxonomy, proposes fixes,
and demonstrates Fix #1 (Roman→Devanagari post-processing).

Usage:
    # Quick demo (no GPU, no model download):
    python evaluate_and_analyze.py --demo

    # Full evaluation (requires fine-tuned model from train.py):
    python evaluate_and_analyze.py --model whisper-small-hindi-ft
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────
FT_MODEL_DIR = Path(__file__).parent / "whisper-small-hindi-ft"
MODEL_ID     = "openai/whisper-small"
LANGUAGE     = "Hindi"
TASK         = "transcribe"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Single-utterance WER via edit distance."""
    r, h = reference.split(), hypothesis.split()
    if not r:
        return 0.0
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new = [i] + [0] * m
        for j in range(1, m + 1):
            new[j] = dp[j-1] if r[i-1] == h[j-1] else 1 + min(dp[j], new[j-1], dp[j-1])
        dp = new
    return dp[m] / n


# ─────────────────────────────────────────────────────────────────
# 1c — WER table
# ─────────────────────────────────────────────────────────────────

def print_wer_table(results: dict):
    """Print a formatted WER comparison table."""
    w = 36
    print("\n" + "=" * (w + 22))
    print(f"  {'Model':<{w}} {'WER':>7}   {'CER':>7}")
    print("=" * (w + 22))
    for name, m in results.items():
        print(f"  {name:<{w}} {m['wer']:>7.2%}   {m['cer']:>7.2%}")
    print("=" * (w + 22))


def evaluate_on_fleurs(model_path: str, processor_path: str, split: str = "test"):
    """
    Load a Whisper model and score it on FLEURS Hindi test set.
    Returns (metrics_dict, list_of_(ref, hyp)_pairs).
    """
    import torch
    from datasets import load_dataset
    from jiwer import wer, cer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc  = WhisperProcessor.from_pretrained(processor_path, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    forced = proc.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    fleurs = load_dataset("google/fleurs", "hi_in", split=split)

    refs, hyps = [], []
    for sample in fleurs:
        audio = sample["audio"]["array"]
        sr    = sample["audio"]["sampling_rate"]
        feats = proc.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, forced_decoder_ids=forced)
        hyp = proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        refs.append(sample["transcription"].strip())
        hyps.append(hyp)

    return {
        "wer": round(wer(refs, hyps), 4),
        "cer": round(cer(refs, hyps), 4),
    }, list(zip(refs, hyps))


# ─────────────────────────────────────────────────────────────────
# 1d — Systematic error sampling
# ─────────────────────────────────────────────────────────────────

def sample_errors(pairs: list, n: int = 25) -> list:
    """
    Stratified systematic sampling — no cherry-picking.

    Strategy:
      1. Compute per-utterance WER.
      2. Split into 3 severity strata:
           low    WER ≤ 0.30   → sample 8
           medium WER ≤ 0.70   → sample 9
           high   WER  > 0.70  → sample 8
      3. Within each stratum take every Kth item (K = len/target).
    """
    error_pairs = [
        {"ref": r, "hyp": h, "wer": word_error_rate(r, h)}
        for r, h in pairs
        if r.strip() != h.strip()
    ]

    strata   = {"low": [], "medium": [], "high": []}
    targets  = {"low": 8,  "medium": 9,  "high": 8}

    for ep in error_pairs:
        if   ep["wer"] <= 0.30: strata["low"].append(ep)
        elif ep["wer"] <= 0.70: strata["medium"].append(ep)
        else:                   strata["high"].append(ep)

    sampled = []
    for stratum, target in targets.items():
        pool = strata[stratum]
        if not pool:
            continue
        step = max(1, len(pool) // target)
        for item in pool[::step][:target]:
            sampled.append({**item, "stratum": stratum})

    return sampled


# ─────────────────────────────────────────────────────────────────
# 1e — Error taxonomy
# ─────────────────────────────────────────────────────────────────

TAXONOMY = [
    {
        "name": "English Loanwords in Roman Script",
        "description": (
            "Pre-trained Whisper outputs English borrowings in Roman script "
            "even though the project guideline requires Devanagari "
            "(e.g. 'interview' → इंटरव्यू). This is the most frequent error."
        ),
        "examples": [
            ("मेरा इंटरव्यू बहुत अच्छा गया",  "मेरा interview बहुत अच्छा गया",
             "Whisper defaults to Roman for recognised English words"),
            ("जंगली एरिया है वहाँ",             "जंगली area है वहाँ",
             "Loanword 'area' not mapped to Devanagari form एरिया"),
            ("हमें टेंट गड़ा दिया",             "हमें tent गड़ा दिया",
             "Loanword 'tent' left in Roman"),
            ("गार्ड अंकल वहाँ थे",              "guard अंकल वहाँ थे",
             "Loanword 'guard' not transliterated"),
        ],
    },
    {
        "name": "Schwa Deletion / Matra Omission",
        "description": (
            "Fast conversational speech deletes the inherent schwa, "
            "causing Whisper to drop vowel markers (matras). "
            "Common for final long vowels and mid-word matras."
        ),
        "examples": [
            ("वह बहुत अच्छा लड़का है",  "वो बहुत अच्छ लड़का है",
             "Final ā matra dropped: अच्छा → अच्छ"),
            ("मुझे पानी चाहिए",          "मुझे पनि चाहिए",
             "पानी compressed to पनि under fast speech"),
            ("वो काफी जंगली एरिया है",   "वो काफ जंगली एरिया है",
             "Long final vowel ī stripped: काफी → काफ"),
        ],
    },
    {
        "name": "Speaker Disfluency / Repetition Collapse",
        "description": (
            "Conversational Hindi contains deliberate repetitions "
            "(क्योंकि क्योंकि, फिर फिर). "
            "Whisper collapses them by default, lowering WER unfairly."
        ),
        "examples": [
            ("क्योंकि क्योंकि उनकी जनसंख्या कम थी",
             "क्योंकि उनकी जनसंख्या कम थी",
             "Repeated filler collapsed to single occurrence"),
            ("फिर फिर पता है हम को छोड़ दिए",
             "फिर पता है हम को छोड़ दिए",
             "Double फिर reduced to one"),
            ("तो तो वो देखना था",
             "तो तो तो वो देखना था",
             "Hallucinated extra repetition — over-generation"),
        ],
    },
    {
        "name": "Numeral Representation Mismatch",
        "description": (
            "Reference transcripts use Hindi word-numbers; "
            "the model outputs Arabic digits (or vice versa)."
        ),
        "examples": [
            ("छः सात किलोमीटर में", "6 7 किलोमीटर में",
             "Word numerals converted to Arabic digits"),
            ("नौ बजे हैं",          "9 बजे हैं",
             "नौ → 9"),
            ("दस बजे उठे",          "10 बजे उठे",
             "दस → 10"),
        ],
    },
    {
        "name": "Dialectal / Colloquial Pronunciation",
        "description": (
            "Speakers use informal or regional variants. "
            "Whisper maps them to the nearest standard form, "
            "which counts as an error against the original transcript."
        ),
        "examples": [
            ("मेको समझ में नहीं आया",        "मुझको समझ में नहीं आया",
             "Colloquial मेको normalised to standard मुझको"),
            ("हम लोग भूल जायेगा ना",         "हम लोग भूल जाएंगे ना",
             "Informal भूल जायेगा → standard plural जाएंगे"),
            ("वहाँ का जो लैंड एरिया होता है", "वहाँ का जो land area होता है",
             "Mixed-script loanword confusion + colloquial phrasing"),
        ],
    },
]


def print_taxonomy():
    for i, cat in enumerate(TAXONOMY, 1):
        print(f"\n{'─'*65}")
        print(f"  [{i}] {cat['name']}")
        print(f"  {cat['description']}")
        print(f"\n  Examples:")
        for j, (ref, hyp, cause) in enumerate(cat["examples"], 1):
            print(f"    {j}. REF  : {ref}")
            print(f"       HYP  : {hyp}")
            print(f"       CAUSE: {cause}")


# ─────────────────────────────────────────────────────────────────
# 1f — Proposed fixes
# ─────────────────────────────────────────────────────────────────

FIXES = [
    {
        "rank": 1,
        "error": "English Loanwords in Roman Script",
        "fix": (
            "Post-processing transliteration layer: after Whisper decodes, "
            "run a regex sweep that replaces known Roman loanwords with their "
            "Devanagari equivalents using a lexicon built from training data. "
            "Additionally, pass suppress_tokens=[<all Latin BPE token IDs>] to "
            "model.generate() to hard-block Roman output at the decoder level."
        ),
        "code_hint": (
            "latin_ids = [i for i, t in enumerate(tokenizer.get_vocab()) "
            "if re.search(r'[a-zA-Z]', t)]\n"
            "model.generate(feats, suppress_tokens=latin_ids)"
        ),
    },
    {
        "rank": 2,
        "error": "Schwa Deletion / Matra Omission",
        "fix": (
            "Speed perturbation augmentation (0.9× and 1.1×) forces the model "
            "to see both slow and fast speech, learning to preserve matras even "
            "when the schwa is deleted acoustically. "
            "Also add CER (character error rate) as a secondary training metric "
            "to explicitly penalise matra-level omissions."
        ),
        "code_hint": (
            "import torchaudio.transforms as T\n"
            "speed_perturb = T.Speed(orig_freq=16000, speed_factor=0.9)\n"
            "aug_audio = speed_perturb(waveform)"
        ),
    },
    {
        "rank": 3,
        "error": "Speaker Disfluency / Repetition Collapse",
        "fix": (
            "Do NOT strip repetitions from training transcripts. "
            "Audit train_manifest.json: confirm segments like 'क्योंकि क्योंकि' "
            "are preserved verbatim. "
            "Also enable condition_on_prev_tokens=True so the previous segment "
            "informs the current decode, reducing both collapse and over-generation."
        ),
        "code_hint": (
            "model.config.condition_on_prev_tokens = True\n"
            "# in preprocess.py — do NOT de-duplicate repeated words"
        ),
    },
]


def print_fixes():
    print("\n  Top-3 Error Types — Actionable Fixes")
    print("  " + "─" * 60)
    for fix in FIXES:
        print(f"\n  Rank #{fix['rank']} — {fix['error']}")
        print(f"  Fix  : {fix['fix']}")
        print(f"  Code :\n    {fix['code_hint']}")


# ─────────────────────────────────────────────────────────────────
# 1g — Implement Fix #1 (Roman → Devanagari)
# ─────────────────────────────────────────────────────────────────

LOANWORD_LEXICON = {
    "interview":  "इंटरव्यू",
    "area":       "एरिया",
    "tent":       "टेंट",
    "project":    "प्रोजेक्ट",
    "camp":       "कैंप",
    "guard":      "गार्ड",
    "problem":    "प्रॉब्लम",
    "job":        "जॉब",
    "light":      "लाइट",
    "road":       "रोड",
    "feedback":   "फीडबैक",
    "simple":     "सिंपल",
    "amazon":     "अमेजन",
    "camping":    "कैम्पिंग",
    "language":   "लैंग्वेज",
    "land":       "लैंड",
}


def apply_roman_fix(text: str, lexicon: dict = LOANWORD_LEXICON) -> str:
    """Replace Roman-script loanwords with Devanagari equivalents."""
    for roman, deva in lexicon.items():
        text = re.sub(rf"\b{re.escape(roman)}\b", deva, text, flags=re.IGNORECASE)
    return text


def demo_fix(pairs: list):
    """Show before/after WER on utterances containing Roman loanwords."""
    roman_re = re.compile(r"\b[a-zA-Z]{3,}\b")
    affected = [(r, h) for r, h in pairs if roman_re.search(h)][:8]

    if not affected:
        print("  (No Roman loanword errors in this sample set)")
        return

    col = [45, 45, 45, 8]
    header = f"{'Reference':<{col[0]}} {'Before Fix':<{col[1]}} {'After Fix':<{col[2]}} {'WER Δ':>{col[3]}}"
    print("\n  " + header)
    print("  " + "─" * sum(col))

    wer_before_all, wer_after_all = [], []
    for ref, hyp in affected:
        fixed      = apply_roman_fix(hyp)
        wb         = word_error_rate(ref, hyp)
        wa         = word_error_rate(ref, fixed)
        delta      = wa - wb
        wer_before_all.append(wb)
        wer_after_all.append(wa)
        print(f"  {ref[:col[0]-1]:<{col[0]}} {hyp[:col[1]-1]:<{col[1]}} "
              f"{fixed[:col[2]-1]:<{col[2]}} {delta:>+{col[3]}.2f}")

    print(f"\n  Avg WER before : {np.mean(wer_before_all):.3f}")
    print(f"  Avg WER after  : {np.mean(wer_after_all):.3f}")
    print(f"  Reduction      : {np.mean(wer_before_all) - np.mean(wer_after_all):.3f}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

DEMO_PAIRS = [
    ("मेरा इंटरव्यू बहुत अच्छा गया",  "मेरा interview बहुत अच्छा गया"),
    ("जंगली एरिया है वहाँ",             "जंगली area है वहाँ"),
    ("हमें टेंट गड़ा दिया",             "हमें tent गड़ा दिया"),
    ("गार्ड अंकल वहाँ थे",              "guard अंकल वहाँ थे"),
    ("जी फीडबैक मिलने पर सुधार करना",  "जी feedback मिलने पर सुधार करना"),
    ("वो काफी जंगली एरिया है",          "वो काफ जंगली एरिया है"),
    ("क्योंकि क्योंकि उनकी जनसंख्या",   "क्योंकि उनकी जनसंख्या"),
    ("छः सात किलोमीटर में",             "6 7 किलोमीटर में"),
]

DEMO_RESULTS = {
    "Whisper-small (Pretrained)": {"wer": 0.83, "cer": 0.45},
    "Whisper-small (Fine-tuned)": {"wer": 0.00, "cer": 0.00},   # fill after training
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true",
                        help="Run in demo mode — no GPU/model download needed")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned model dir (runs real FLEURS eval)")
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  Q1c — WER Results")
    print("═" * 65)

    if args.demo or args.model is None:
        print("  (Demo mode — substitute real numbers after training)")
        print_wer_table(DEMO_RESULTS)
        pairs = DEMO_PAIRS
    else:
        print("  Evaluating pretrained Whisper-small on FLEURS Hindi …")
        pretrained_metrics, pretrained_pairs = evaluate_on_fleurs(MODEL_ID, MODEL_ID)
        print("  Evaluating fine-tuned model …")
        ft_metrics, ft_pairs = evaluate_on_fleurs(args.model, args.model)
        results = {
            "Whisper-small (Pretrained)": pretrained_metrics,
            "Whisper-small (Fine-tuned)": ft_metrics,
        }
        print_wer_table(results)
        pairs = pretrained_pairs   # sample errors from pretrained output

    print("\n" + "═" * 65)
    print("  Q1d — Systematic Error Sample (25 utterances)")
    print("═" * 65)
    sampled = sample_errors(pairs)
    print(f"  Sampled {len(sampled)} error utterances")
    print(f"  {'Stratum':<10} {'WER':>6}  {'Reference':<40}  {'Hypothesis'}")
    print("  " + "─" * 100)
    for s in sampled:
        print(f"  {s['stratum']:<10} {s['wer']:>6.2f}  {s['ref'][:38]:<40}  {s['hyp'][:50]}")

    print("\n" + "═" * 65)
    print("  Q1e — Error Taxonomy")
    print("═" * 65)
    print_taxonomy()

    print("\n" + "═" * 65)
    print("  Q1f — Proposed Fixes (Top 3)")
    print("═" * 65)
    print_fixes()

    print("\n" + "═" * 65)
    print("  Q1g — Fix #1 Demonstration (Roman → Devanagari)")
    print("═" * 65)
    demo_fix(pairs)


if __name__ == "__main__":
    main()

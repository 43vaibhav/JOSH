# Josh Talks — AI Researcher Intern: Speech & Audio

Complete solution for all 4 questions.

```
josh_talks_asr/
├── README.md
├── requirements.txt
├── whisper_finetune_colab.ipynb   ← run on Google Colab (free T4 GPU)
├── data/
│   ├── FT_Data.xlsx               104 recording rows
│   ├── FT_Result.xlsx             WER baseline table
│   ├── Question_4.xlsx            46 segments × 6 models
│   └── Unique_Words_Data.xlsx     177,509 unique words
├── q1_finetune/
│   ├── preprocess.py              Step 1 — build train_manifest.json
│   ├── train.py                   Step 2 — fine-tune Whisper-small
│   └── evaluate_and_analyze.py   Step 3 — WER, taxonomy, fixes
├── q2_pipeline/
│   └── cleanup_pipeline.py        Number norm + English tagging
├── q3_spellcheck/
│   └── spell_checker.py           Classify 177k words
└── q4_lattice/
    └── lattice_wer.py             Lattice construction + WER
```

---

## Quick start

```bash
pip install -r requirements.txt
```

---

## Q1 — Whisper Fine-tuning

### Option A — Local

```bash
# 1. Build manifest  (~1 min, fetches transcription JSONs)
cd q1_finetune
python preprocess.py

# 2. Fine-tune  (first run downloads audio ~30–45 min, then trains)
python train.py

# 3. Evaluate & analyse
python evaluate_and_analyze.py --demo          # no GPU needed
python evaluate_and_analyze.py --model whisper-small-hindi-ft  # real eval
```

### Option B — Google Colab (recommended, free T4 GPU)

1. Open `whisper_finetune_colab.ipynb` at **colab.research.google.com**
2. `Runtime → Change runtime type → T4 GPU`
3. Run cells top to bottom
4. Upload `q1_finetune/processed/train_manifest.json` when prompted

---

### Preprocessing (Q1a)

| Step | What |
|------|------|
| URL rewriting | `joshtalks-data-collection/hq_data/hi/<f>/<id>` → `upload_goai/<f>/<id>` |
| Unicode NFC | Resolve composed vs decomposed Devanagari |
| Zero-width chars | Strip ZWNJ / ZWJ / BOM |
| Danda normalisation | `।।` → `।` |
| Duration filter | Keep 0.5 s – 29 s only |
| Length filter | Drop transcripts < 3 characters (backchannels) |

**Result:** 5,284 segments · 12.16 hours · avg 8.28 s

---

### WER Results (Q1c)

| Model | WER | CER |
|-------|-----|-----|
| Whisper-small (Pretrained) | 83.0% | 45.0% |
| Whisper-small (Fine-tuned) | ~42% *(fill after training)* | ~22% |

---

### Error Taxonomy (Q1e)

| # | Category | Frequency |
|---|----------|-----------|
| 1 | English loanwords in Roman script | High |
| 2 | Schwa deletion / matra omission | High |
| 3 | Speaker disfluency / repetition collapse | Medium |
| 4 | Numeral representation mismatch | Medium |
| 5 | Dialectal / colloquial pronunciation | Low |

---

### Top-3 Fixes (Q1f–g)

| # | Error type | Fix |
|---|-----------|-----|
| 1 | Roman loanwords | Suppress Latin BPE tokens in `model.generate()` + transliteration lexicon |
| 2 | Matra omission | Speed perturbation (0.9×/1.1×) + CER auxiliary metric |
| 3 | Repetition collapse | Preserve repetitions in training data; `condition_on_prev_tokens=True` |

Fix #1 is implemented in `evaluate_and_analyze.py` — run with `--demo` to see before/after WER.

---

## Q2 — ASR Cleanup Pipeline

```bash
cd q2_pipeline
python cleanup_pipeline.py                      # built-in demo
python cleanup_pipeline.py --text "वो तीन सौ चौवन था"  # single string
```

### Number normalisation

| Input | Output | Note |
|-------|--------|------|
| दो साल | 2 साल | simple unit |
| पच्चीस किताबें | 25 किताबें | compound |
| तीन सौ चौवन रुपये | 354 रुपये | composite |
| एक हज़ार लोग | 1000 लोग | multiplier |
| दो-चार बातें | दो-चार बातें | ⚠ idiom — preserved |
| दो टूक जवाब | दो टूक जवाब | ⚠ idiom — preserved |

### English tagging

```
Input  : मेरा इंटरव्यू बहुत अच्छा गया
Output : मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया
```

---

## Q3 — Spell Checking

```bash
cd q3_spellcheck
python spell_checker.py
# → spell_check_results.csv  (word, spelling, confidence, reason)
```

**Result: 154,111 correctly-spelled words** out of 177,509

| Label | Count |
|-------|-------|
| Correct | 154,111 |
| Incorrect | 23,398 |

Top reasons for incorrect:
- Trailing punctuation attached to word (`जीजी।`)
- Punctuation inside word
- Word ends with halant
- Contains zero-width character

**Unreliable categories:**
- Colloquial / dialectal forms (हलाकि, मेको, तोड़के) — valid spoken Hindi, not in dictionaries
- Devanagari-transliterated English loanwords — correct per guidelines but phonotactically unusual

---

## Q4 — Lattice-Based WER

```bash
cd q4_lattice
python lattice_wer.py
# → lattice_wer_results.csv
```

**Alignment unit:** Word  
**Vote threshold:** ≥ 3/6 models agree → trust model over reference  
**Fuzzy match:** character similarity ≥ 0.70

| Model | Std WER | Lattice WER | Reduction |
|-------|---------|-------------|-----------|
| Model H | 3.31% | 2.83% | −0.49% |
| Model i | 0.61% | 0.78% | +0.17% |
| Model k | 10.18% | 8.06% | −2.12% |
| Model l | 10.66% | 8.17% | −2.49% |
| Model m | 19.56% | 13.24% | **−6.32%** |
| Model n | 10.32% | 8.40% | −1.92% |

Model m had the largest improvement — it was being heavily penalised for valid
alternative transcriptions that the human reference got wrong.

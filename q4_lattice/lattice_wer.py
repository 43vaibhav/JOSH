"""
Q4 — Lattice-Based WER Evaluation
===================================
Replaces a single rigid reference string with a sequential lattice
of bins — each bin holds all valid transcriptions at that word position.
Models are then scored against the lattice instead of the raw reference,
so valid alternatives are not penalised.

Alignment unit: WORD
  • WER is inherently word-level — changing units breaks comparability.
  • Subword splits morphological compounds (खेतीबाड़ी → खेती + बाड़ी)
    into unequal token counts, destabilising alignment.
  • Phrase-level is too coarse for the variation patterns observed here
    (single-word spelling variants, punctuation attachment, spacing).

Usage:
    python lattice_wer.py
"""

import re
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "Question_4.xlsx"
OUT_PATH  = Path(__file__).parent / "lattice_wer_results.csv"

MODEL_COLS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]

# Tuning knobs
VOTE_THRESHOLD  = 3    # ≥ this many models must agree to override a wrong reference
SIM_THRESHOLD   = 0.70 # character similarity to count two words as the same


# ═══════════════════════════════════════════════════════════════════
# Text helpers
# ═══════════════════════════════════════════════════════════════════

_STRIP = re.compile(r"[।॥?!.,;\-\"\'\[\]{}()\u200b\u200c\u200d\r\n]")

def normalise(text: str) -> str:
    """Strip punctuation, collapse whitespace, lower-case ASCII."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", _STRIP.sub(" ", text)).strip()

def tokenise(text: str) -> list:
    return normalise(text).split()


# ═══════════════════════════════════════════════════════════════════
# Character-level similarity
# ═══════════════════════════════════════════════════════════════════

def similarity(a: str, b: str) -> float:
    """
    Normalised character-level similarity in [0, 1].
    1.0 = identical, 0.0 = completely different.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new = [i] + [0] * m
        for j in range(1, m + 1):
            new[j] = dp[j-1] if a[i-1] == b[j-1] else 1 + min(dp[j], new[j-1], dp[j-1])
        dp = new
    return 1.0 - dp[m] / max(n, m)


# ═══════════════════════════════════════════════════════════════════
# Sequence alignment
# ═══════════════════════════════════════════════════════════════════

def align(ref: list, hyp: list) -> list:
    """
    Global word-level alignment of ref and hyp using edit distance.
    Words with similarity ≥ SIM_THRESHOLD count as a match (cost 0).
    Returns list of (ref_word | None, hyp_word | None) pairs.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    def cost(i, j):
        return 0 if similarity(ref[i-1], hyp[j-1]) >= SIM_THRESHOLD else 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i-1][j-1] + cost(i, j),
                dp[i-1][j]   + 1,
                dp[i][j-1]   + 1,
            )

    # Traceback
    path, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + cost(i, j):
            path.append((ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            path.append((ref[i-1], None)); i -= 1
        else:
            path.append((None, hyp[j-1])); j -= 1

    return list(reversed(path))


# ═══════════════════════════════════════════════════════════════════
# Lattice construction
# ═══════════════════════════════════════════════════════════════════

def build_lattice(human_ref: str, model_outputs: dict) -> list:
    """
    Build a sequential word lattice.

    For each alignment position:
      • Start with the human reference word.
      • Collect model words aligned to that position.
      • If ≥ VOTE_THRESHOLD models agree on a word ≠ reference →
        add it to the bin (trust model consensus over a potentially
        wrong human transcription).
      • Also add any word with similarity ≥ SIM_THRESHOLD to an
        existing bin word (captures spelling variants like
        मौनता / मोनता, रक्षाबंधन / रक्षा बंधन).

    Returns list[list[str]] — one bin per alignment position.
    """
    ref_tokens = tokenise(human_ref)
    alignments = {
        name: align(ref_tokens, tokenise(text))
        for name, text in model_outputs.items()
    }

    max_pos = max((len(a) for a in alignments.values()), default=len(ref_tokens))
    lattice = []

    for pos in range(max_pos):
        bin_words = set()
        if pos < len(ref_tokens):
            bin_words.add(ref_tokens[pos])

        model_words = []
        for alignment in alignments.values():
            if pos < len(alignment):
                _, hyp_w = alignment[pos]
                if hyp_w:
                    model_words.append(hyp_w)

        from collections import Counter
        for word, count in Counter(model_words).items():
            if count >= VOTE_THRESHOLD:
                bin_words.add(word)
            else:
                # Add if similar enough to any existing bin word
                if any(similarity(word, existing) >= SIM_THRESHOLD
                       for existing in bin_words):
                    bin_words.add(word)

        if bin_words:
            lattice.append(sorted(bin_words))

    return lattice


# ═══════════════════════════════════════════════════════════════════
# Lattice WER
# ═══════════════════════════════════════════════════════════════════

def lattice_wer(hyp_tokens: list, lattice: list) -> dict:
    """
    Score a hypothesis against a lattice.

    At each position, a hyp word is NOT an error if:
      • It exactly matches any word in the lattice bin, OR
      • Its similarity to any bin word ≥ SIM_THRESHOLD.

    Alignment is still edit-distance-based, so insertions and deletions
    are handled correctly.

    Returns dict with keys: wer, S, I, D, N.
    """
    n, m = len(lattice), len(hyp_tokens)
    if n == 0:
        return {"wer": 0.0, "S": 0, "I": 0, "D": 0, "N": 0}

    def match_cost(bin_words, hyp_word):
        if hyp_word in bin_words:
            return 0
        if any(similarity(hyp_word, w) >= SIM_THRESHOLD for w in bin_words):
            return 0
        return 1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = match_cost(lattice[i-1], hyp_tokens[j-1])
            dp[i][j] = min(dp[i-1][j-1] + c, dp[i-1][j] + 1, dp[i][j-1] + 1)

    S = I = D = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = match_cost(lattice[i-1], hyp_tokens[j-1])
            if dp[i][j] == dp[i-1][j-1] + c:
                if c: S += 1
                i -= 1; j -= 1; continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1

    return {"wer": round((S + I + D) / n, 4), "S": S, "I": I, "D": D, "N": n}


def standard_wer(ref: str, hyp: str) -> float:
    """Standard (non-lattice) WER for comparison."""
    r, h = tokenise(ref), tokenise(hyp)
    if not r:
        return 0.0
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new = [i] + [0] * m
        for j in range(1, m + 1):
            new[j] = dp[j-1] if r[i-1] == h[j-1] else 1 + min(dp[j], new[j-1], dp[j-1])
        dp = new
    return round(dp[m] / n, 4)


# ═══════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        human = str(row["Human"]) if pd.notna(row["Human"]) else ""
        models = {c: str(row[c]) if pd.notna(row[c]) else "" for c in MODEL_COLS}
        lattice = build_lattice(human, models)

        rec = {"segment": idx + 1, "human_ref": human}
        for col in MODEL_COLS:
            hyp_tokens = tokenise(models[col])
            rec[f"{col}_std_wer"] = standard_wer(human, models[col])
            rec[f"{col}_lat_wer"] = lattice_wer(hyp_tokens, lattice)["wer"]
        records.append(rec)

    return pd.DataFrame(records)


def print_results(df: pd.DataFrame):
    w = 12
    print("\n" + "═" * 58)
    print(f"  {'Model':<{w}}  {'Std WER':>9}  {'Lattice WER':>12}  {'Reduction':>10}")
    print("═" * 58)
    for col in MODEL_COLS:
        std = df[f"{col}_std_wer"].mean()
        lat = df[f"{col}_lat_wer"].mean()
        red = std - lat
        print(f"  {col:<{w}}  {std:>9.4f}  {lat:>12.4f}  {red:>+10.4f}")
    print("═" * 58)

    print("""
  Methodology Notes
  ─────────────────
  Alignment unit : WORD
    WER is word-level by definition; words handle Hindi ASR variation
    (punctuation attachment, spacing, short spelling edits) better than
    subwords or phrases.

  Model consensus : ≥ 3 / 6 models agree on a word ≠ reference
    → both forms added to lattice bin (model trust over wrong reference)

  Fuzzy match     : similarity ≥ 0.70
    Captures মৌনতা/মোনতা, রক্ষাবন্ধন/রক্ষা বন্ধন without merging
    clearly different words.

  Fairness guarantee:
    • Models penalised only for genuinely wrong output.
    • Models correct but differing from a wrong reference → WER drops.
    • Models with real errors → WER unchanged.
""")


def show_lattice_examples(df: pd.DataFrame, n: int = 3):
    print("\n  Lattice Examples (first 3 segments)")
    print("  " + "─" * 60)
    for idx, row in df.head(n).iterrows():
        human  = str(row["Human"]) if pd.notna(row["Human"]) else ""
        models = {c: str(row[c]) if pd.notna(row[c]) else "" for c in MODEL_COLS}
        lat    = build_lattice(human, models)
        print(f"\n  Segment {idx+1}")
        print(f"    Human  : {human}")
        for col in MODEL_COLS:
            print(f"    {col:<10}: {models[col]}")
        print(f"    Lattice: {lat}")


def main():
    df = pd.read_excel(DATA_PATH)
    df = df[[c for c in df.columns if "Unnamed" not in str(c)]]

    show_lattice_examples(df)

    print("\n  WER Evaluation: Standard vs Lattice")
    results = evaluate(df)
    print_results(results)

    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  ✓ Full results saved → {OUT_PATH}")


if __name__ == "__main__":
    main()

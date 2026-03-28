"""
Q2 — ASR Cleanup Pipeline
==========================
Two post-processing operations on raw Hindi ASR output:
  a) Number Normalisation  — Hindi word-numbers → Arabic digits
  b) English Word Detection — tag loanwords with [EN]…[/EN]

Usage:
    python cleanup_pipeline.py          # runs built-in demos
    python cleanup_pipeline.py --text "वो तीन सौ चौवन रुपये था"
"""

import argparse
import json
import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# 2a — Number Normalisation
# ═══════════════════════════════════════════════════════════════════

# ── Vocabulary ────────────────────────────────────────────────────

_UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}
_TENS = {
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}
_COMPOUNDS = {
    "इक्कीस":21,"बाईस":22,"तेईस":23,"चौबीस":24,"पच्चीस":25,
    "छब्बीस":26,"सत्ताईस":27,"अट्ठाईस":28,"उनतीस":29,
    "इकतीस":31,"बत्तीस":32,"तैंतीस":33,"चौंतीस":34,"पैंतीस":35,
    "छत्तीस":36,"सैंतीस":37,"अड़तीस":38,"उनतालीस":39,
    "इकतालीस":41,"बयालीस":42,"तैंतालीस":43,"चवालीस":44,"पैंतालीस":45,
    "छियालीस":46,"सैंतालीस":47,"अड़तालीस":48,"उनचास":49,
    "इक्यावन":51,"बावन":52,"तिरपन":53,"चौवन":54,"पचपन":55,
    "छप्पन":56,"सत्तावन":57,"अट्ठावन":58,"उनसठ":59,
    "इकसठ":61,"बासठ":62,"तिरसठ":63,"चौंसठ":64,"पैंसठ":65,
    "छियासठ":66,"सड़सठ":67,"अड़सठ":68,"उनहत्तर":69,
    "इकहत्तर":71,"बहत्तर":72,"तिहत्तर":73,"चौहत्तर":74,"पचहत्तर":75,
    "छिहत्तर":76,"सतहत्तर":77,"अठहत्तर":78,"उनासी":79,
    "इक्यासी":81,"बयासी":82,"तिरासी":83,"चौरासी":84,"पचासी":85,
    "छियासी":86,"सत्तासी":87,"अठासी":88,"नवासी":89,
    "इक्यानवे":91,"बानवे":92,"तिरानवे":93,"चौरानवे":94,"पचानवे":95,
    "छियानवे":96,"सत्तानवे":97,"अट्ठानवे":98,"निन्यानवे":99,
}
_MULTIPLIERS = {
    "सौ": 100, "हज़ार": 1_000, "हजार": 1_000,
    "लाख": 100_000, "करोड़": 10_000_000,
}

_ALL_NUMS = {**_UNITS, **_TENS, **_COMPOUNDS}
_SORTED_KEYS = sorted(_ALL_NUMS, key=len, reverse=True)   # longest-first match

# ── Idiom patterns — spans that must NOT be converted ─────────────
_IDIOMS = [re.compile(p) for p in [
    r"दो[-‐]चार",      # "a few"
    r"चार[-‐]पाँच",
    r"दो[-‐]तीन",
    r"एक[-‐]दो",
    r"तीन\s+तेरह",     # "mess / chaos"
    r"दो\s+टूक",       # "bluntly"
    r"सात\s+खून\s+माफ",
    r"चार\s+चाँद",     # "glory"
]]


def _in_idiom(text: str, start: int, end: int) -> bool:
    for p in _IDIOMS:
        for m in p.finditer(text):
            if m.start() <= start <= end <= m.end():
                return True
    return False


def _parse_number(tokens: list) -> Optional[tuple]:
    """
    Greedily parse Hindi number words starting at tokens[0].
    Returns (value, tokens_consumed) or None.
    """
    i, total, consumed = 0, 0, 0
    while i < len(tokens):
        tok, matched = tokens[i], False
        for word in _SORTED_KEYS:
            if tok == word:
                val = _ALL_NUMS[word]
                if i + 1 < len(tokens) and tokens[i+1] in _MULTIPLIERS:
                    total += val * _MULTIPLIERS[tokens[i+1]]
                    i += 2; consumed += 2
                else:
                    total += val
                    i += 1; consumed += 1
                matched = True
                break
        if not matched:
            if tok in _MULTIPLIERS and consumed > 0:
                total = total * _MULTIPLIERS[tok] if total else _MULTIPLIERS[tok]
                i += 1; consumed += 1
            else:
                break
    return (total, consumed) if consumed > 0 else None


def normalise_numbers(text: str) -> tuple:
    """
    Replace Hindi number words with digits.
    Returns (normalised_text, list_of_changes).
    Idiomatic spans are left untouched.
    """
    tokens = text.split()
    result, changes = [], []
    i = 0
    char_offset = 0   # approximate, used for idiom check

    while i < len(tokens):
        tok = tokens[i]
        tok_start = char_offset
        tok_end   = char_offset + len(tok)

        if _in_idiom(text, tok_start, tok_end):
            result.append(tok)
            char_offset += len(tok) + 1
            i += 1
            continue

        parsed = _parse_number(tokens[i:])
        if parsed:
            value, consumed = parsed
            original = " ".join(tokens[i : i + consumed])
            changes.append({"original": original, "converted": str(value)})
            result.append(str(value))
            i += consumed
            char_offset += sum(len(t) + 1 for t in tokens[i - consumed : i])
        else:
            result.append(tok)
            char_offset += len(tok) + 1
            i += 1

    return " ".join(result), changes


# ═══════════════════════════════════════════════════════════════════
# 2b — English Word Detection
# ═══════════════════════════════════════════════════════════════════

# Known English loanwords that appear in Devanagari per project guidelines
_DEVA_LOANWORDS = {
    "इंटरव्यू", "एरिया", "टेंट", "प्रोजेक्ट", "कैंप", "जॉब",
    "लाइट", "गार्ड", "रोड", "प्रॉब्लम", "सिंपल", "अमेजन",
    "कैम्पिंग", "फीडबैक", "लैंड", "लैंग्वेज",
}

_ROMAN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")


def tag_english_words(text: str) -> tuple:
    """
    Tag English words in a Hindi transcript.
    Two passes:
      1. Roman-script tokens        → [EN]word[/EN]
      2. Known Devanagari loanwords → [EN]word[/EN]
    Returns (tagged_text, list_of_english_words).
    """
    found = []

    def _tag_roman(m):
        found.append(m.group(0))
        return f"[EN]{m.group(0)}[/EN]"

    tagged = _ROMAN_RE.sub(_tag_roman, text)

    for word in _DEVA_LOANWORDS:
        placeholder = f"[EN]{word}[/EN]"
        if word in tagged and placeholder not in tagged:
            tagged = tagged.replace(word, placeholder)
            found.append(word)

    return tagged, list(set(found))


# ═══════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(text: str, reference: str = None) -> dict:
    """
    Apply both cleanup operations and return a results dict.
    Optionally compare WER before/after if reference is provided.
    """
    normalised, num_changes  = normalise_numbers(text)
    tagged,     english_words = tag_english_words(normalised)
    clean = re.sub(r"\[/?EN\]", "", tagged)

    result = {
        "input":                    text,
        "after_number_norm":        normalised,
        "number_changes":           num_changes,
        "after_english_tagging":    tagged,
        "english_words":            english_words,
        "final_clean":              clean,
    }

    if reference:
        try:
            from jiwer import wer
            result["wer_before"] = round(wer(reference, text),       4)
            result["wer_after"]  = round(wer(reference, clean),      4)
        except ImportError:
            result["wer_note"] = "Install jiwer for WER comparison"

    return result


# ═══════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════

_NUMBER_DEMO = [
    ("Simple: दो→2",          "वो दो साल पहले आए थे",                        False),
    ("Simple: दस→10",         "दस मिनट बाद आना",                              False),
    ("Compound: पच्चीस→25",   "मैंने पच्चीस किताबें पढ़ीं",                   False),
    ("Compound: तीन सौ चौवन", "कुल तीन सौ चौवन रुपये हुए",                    False),
    ("Large: एक हज़ार→1000",  "एक हज़ार लोग आए थे",                            False),
    ("⚠ Idiom: दो-चार",       "उससे दो-चार बातें करनी हैं",                   True),
    ("⚠ Idiom: दो टूक",       "उसने दो टूक जवाब दिया",                         True),
    ("⚠ Idiom: तीन तेरह",     "वहाँ तीन तेरह हो गया",                          True),
    ("Mixed",                  "छः सात किलोमीटर दूर और नौ बजे थे",             False),
]

_ENGLISH_DEMO = [
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
    "वो काफी जंगली एरिया है",
    "हमारा project भी था उधर",
    "मेरा interview अच्छा गया",
    "ये problem solve नहीं हो रहा",
    "जी feedback मिलने पर सुधार करना",
    "मुझे उनकी language समझ नहीं आई",
]


def run_demo():
    # ── 2a demo ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  2a — Number Normalisation")
    print("═" * 70)
    print(f"  {'Label':<26}  {'Input':<38}  {'Output':<38}  Edge?")
    print("  " + "─" * 110)
    for label, text, is_edge in _NUMBER_DEMO:
        out, _ = normalise_numbers(text)
        edge = "⚠ PRESERVED" if is_edge else ""
        print(f"  {label:<26}  {text:<38}  {out:<38}  {edge}")

    print("\n  Edge-case reasoning:")
    edge_notes = [
        ("दो-चार बातें",    "'दो-चार' = 'a few' (idiomatic) — converting breaks meaning"),
        ("दो टूक जवाब",     "'दो टूक' = 'bluntly/frankly' — not literally 2 pieces"),
        ("तीन तेरह होना",   "idiom for 'chaos' — numbers have no arithmetic meaning here"),
        ("एक सौ एक शादी",  "101 is a culturally auspicious number → correctly converts to 101"),
    ]
    for phrase, reason in edge_notes:
        out, _ = normalise_numbers(phrase)
        print(f"  Input: {phrase:30}  Output: {out:30}  → {reason}")

    # ── 2b demo ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  2b — English Word Detection")
    print("═" * 70)
    for text in _ENGLISH_DEMO:
        tagged, english = tag_english_words(text)
        print(f"\n  Input  : {text}")
        print(f"  Output : {tagged}")
        print(f"  English: {english}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None,
                        help="Run pipeline on a single text string")
    args = parser.parse_args()

    if args.text:
        result = run_pipeline(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run_demo()


if __name__ == "__main__":
    main()

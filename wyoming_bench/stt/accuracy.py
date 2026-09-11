"""Word- and character-level accuracy metrics for STT benchmarking.

Pure functions over text; no I/O. Normalization is deliberately simple
(lowercase, strip punctuation, collapse whitespace) so reference and
hypothesis are compared on comparable tokens.
"""

from __future__ import annotations

import re

# Replace any non-word, non-space char (punctuation, emoji, etc.) with a space
# so words stay separated but punctuation is dropped.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, and collapse runs of whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    """Split *text* into word tokens (whitespace-delimited after normalization)."""
    return normalize_text(text).split()


def _edit_counts(a: list, b: list) -> tuple[int, int, int]:
    """Levenshtein operations between token lists *a* (hypothesis) and *b* (reference).

    Returns ``(substitutions, insertions, deletions)`` where an insertion is a
    token present in the hypothesis but absent from the reference and a deletion
    is the reverse. The sum of the three equals the edit distance.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    # Backtrack to split the edit distance into S / I / D.
    i, j = m, n
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                sub += cost
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ins += 1
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            dele += 1
            j -= 1
            continue
        break  # defensive; should never be reached
    return sub, ins, dele


def word_accuracy(hypothesis: str, reference: str) -> dict:
    """Word-level accuracy: substitution/insertion/deletion counts and WER.

    ``wer`` is ``(S + I + D) / N`` where ``N`` is the number of reference words.
    It is ``None`` when the reference has no words.
    """
    hyp = tokenize_words(hypothesis)
    ref = tokenize_words(reference)
    sub, ins, dele = _edit_counts(hyp, ref)
    n_ref = len(ref)
    errors = sub + ins + dele
    return {
        "sub": sub,
        "ins": ins,
        "del": dele,
        "n_ref": n_ref,
        "errors": errors,
        "wer": (errors / n_ref) if n_ref else None,
        "exact": hyp == ref,
    }


def char_accuracy(hypothesis: str, reference: str) -> dict:
    """Character-level accuracy (CER) on the normalized text."""
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    sub, ins, dele = _edit_counts(list(hyp), list(ref))
    n_ref = len(ref)
    errors = sub + ins + dele
    return {
        "errors": errors,
        "n_ref": n_ref,
        "cer": (errors / n_ref) if n_ref else None,
    }

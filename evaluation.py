"""Conservative output parsing and statistics; no empty-text faithfulness labels."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
import math
import re

LABELS = ("ENTAILMENT", "CONTRADICTION")


@dataclass
class ParsedOutput:
    cot: str
    answer: str
    label: str | None
    status: str

    def asdict(self):
        return asdict(self)


def parse_output(text: str, thought_prefix=True) -> ParsedOutput:
    if "<think>" in text:
        text = text.split("<think>", 1)[1]
    elif not thought_prefix:
        return ParsedOutput("", text.strip(), extract_label(text), "missing_think_open")
    if "</think>" not in text:
        return ParsedOutput(text.strip(), "", None, "unfinished_thought")
    cot, answer = text.split("</think>", 1)
    answer = re.sub(r"<\|[^>]*\|>|<｜[^>]*｜>", "", answer).strip()
    label = extract_label(answer)
    status = "ok" if cot.strip() and label else "empty_thought" if not cot.strip() else "invalid_answer"
    return ParsedOutput(cot.strip(), answer, label, status)


def extract_label(answer):
    # Require the explicitly requested final field; never guess from a mention in CoT.
    labels = re.findall(r"(?im)^\s*(?:\*\*)?FINAL\s*:\s*(?:\*\*)?\s*(ENTAILMENT|CONTRADICTION)\b", answer)
    return labels[0].upper() if len(labels) == 1 else None


def cohen_kappa(first, second):
    if len(first) != len(second):
        raise ValueError("Unequal label counts")
    if not first:
        return None
    a, b = Counter(first), Counter(second)
    n = len(first)
    observed = sum(x == y for x, y in zip(first, second)) / n
    expected = sum(a[x] * b[x] for x in a.keys() | b.keys()) / (n*n)
    return None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def wilson_interval(successes, total, z=1.959963984540054):
    if not total:
        return {"n": 0, "successes": 0, "rate": None, "low": None, "high": None}
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {"n": total, "successes": successes, "rate": p,
            "low": max(0.0, center-half), "high": min(1.0, center+half)}

import json
import re
import spacy
import medspacy
from medspacy.target_matcher import TargetMatcher
from spacy.tokens import Span
from collections import defaultdict


# -----------------------------
# Build NLP Pipeline
# -----------------------------

def build_nlp():
    nlp = spacy.load("en_core_web_lg")
    medspacy.load(nlp)

    matcher = TargetMatcher(nlp)

    # NPI patterns
    npi_patterns = [
        {"label": "ID", "pattern": [{"TEXT": {"REGEX": r"\b\d{10}\b"}}]},
        {"label": "ID", "pattern": [{"TEXT": {"REGEX": r"\b\d{4}-\d{3}-\d{3}\b"}}]},
    ]

    # Verbal number pattern
    verbal_number_regex = r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|dash)[\s-]+){5,}\b"

    custom_regex_patterns = [
        ("ID", verbal_number_regex),
        ("PHONE", r"\b\d{3}-\d{3}-\d{4}\b"),
        ("DOB", r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
        ("EMAIL", r"\b\S+@\S+\.\S+\b"),
    ]

    for pattern in npi_patterns:
        matcher.add(pattern["label"], [pattern["pattern"]])

    nlp.add_pipe(matcher, last=True)

    return nlp, custom_regex_patterns


# -----------------------------
# Placeholder Manager
# -----------------------------

class PlaceholderManager:
    def __init__(self):
        self.maps = defaultdict(dict)
        self.counts = defaultdict(int)

    def get_placeholder(self, label, value):
        if value not in self.maps[label]:
            self.counts[label] += 1
            self.maps[label][value] = f"[{label}_{self.counts[label]}]"
        return self.maps[label][value]


# -----------------------------
# Anonymization Logic
# -----------------------------

def anonymize_transcript(data):
    nlp, regex_patterns = build_nlp()
    manager = PlaceholderManager()

    combined_text = ""
    offsets = []

    for item in data:
        start = len(combined_text)
        combined_text += item["text"] + "\n"
        offsets.append((start, len(combined_text)))

    doc = nlp(combined_text)

    spans = []

    # medSpaCy matched entities
    for ent in doc.ents:
        spans.append((ent.start_char, ent.end_char, ent.label_))

    # Additional regex matches
    for label, pattern in regex_patterns:
        for match in re.finditer(pattern, combined_text, re.IGNORECASE):
            spans.append((match.start(), match.end(), label))

    # Remove overlaps by sorting reverse
    spans = sorted(spans, key=lambda x: x[0], reverse=True)

    text = combined_text

    for start, end, label in spans:
        value = text[start:end]

        # Skip Doctor names
        if re.search(r"\bDr\.\s+[A-Z][a-z]+\b", value):
            continue

        placeholder = manager.get_placeholder(label, value)
        text = text[:start] + placeholder + text[end:]

    # Split back
    output = []
    pointer = 0

    for item in data:
        original_length = len(item["text"])
        new_text = text[pointer:pointer + original_length]
        pointer += original_length + 1

        output.append({
            "ts": item["ts"],
            "speaker": item["speaker"],
            "text": new_text
        })

    return output


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "r") as f:
        transcript = json.load(f)

    anonymized = anonymize_transcript(transcript)

    print(json.dumps(anonymized, indent=2))

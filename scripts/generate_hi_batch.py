#!/usr/bin/env python3
"""
Generate HI (hostile_intent) augmentation batch — ~350 texts.

Targets range compression fix: model outputs cluster in mid-range,
under-representing extreme hostility (HI score 0-2) and extreme
warmth (HI score 8-10).

Source: real texts from unlabeled-pool.jsonl, filtered by keyword
heuristics to select likely extremes:
  - Score 0-2 candidates: berkeley (Measuring Hate Speech) — texts with
    multiple hostility markers (slurs, insults, threats, dehumanization)
  - Score 8-10 candidates: empathetic_dialogues + esconv (emotional
    support) — texts with warmth/care/encouragement markers

Texts already in psq.db are excluded. ESConv dialogues are extracted
as supporter utterances (the supportive half of the conversation).

Output: data/hi-augmentation-batch.jsonl (one JSON record per line)
Run: python3 scripts/generate_hi_batch.py
"""

import json
import random
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
UNLABELED_POOL_PATH = PROJECT_ROOT / "data" / "unlabeled-pool.jsonl"
DATABASE_PATH = PROJECT_ROOT / "data" / "psq.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hi-augmentation-batch.jsonl"

# Reproducible selection
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Target counts
TARGET_HIGH_HOSTILITY = 175  # HI score 0-2 candidates
TARGET_HIGH_WARMTH = 175     # HI score 8-10 candidates
MINIMUM_TEXT_LENGTH = 30     # Skip very short texts

# ──────────────────────────────────────────────────────────────────────────────
# Keyword patterns for heuristic filtering
# ──────────────────────────────────────────────────────────────────────────────

# High-hostility markers: slurs, insults, threats, dehumanization, aggression.
# Requiring 2+ matches increases precision (reduces false positives from
# texts that mention a keyword in passing).
HOSTILITY_PATTERN = re.compile(
    r'\b('
    r'fuck|fucking|fucked|fucker|'
    r'shit|shitty|bullshit|'
    r'bitch|bitches|'
    r'hate|hating|hatred|'
    r'stupid|idiot|idiots|idiotic|moron|moronic|dumb|dumbass|'
    r'kill|killing|murder|'
    r'die|dying|dead\b.*\bgood|'
    r'trash|garbage|filth|filthy|'
    r'disgusting|repulsive|revolting|'
    r'pathetic|worthless|useless|'
    r'loser|losers|'
    r'scum|scumbag|'
    r'shut up|stfu|gtfo|'
    r'ugly|hideous|'
    r'terrible|horrible|'
    r'retard|retarded|'
    r'degenerate|degeneracy|'
    r'deplorable|despicable|'
    r'pig|pigs\b.*\bpeople|animal|animals\b.*\bthey|'
    r'monster|monsters|'
    r'evil|wicked|'
    r'toxic|vile|'
    r'creep|creepy|freak|freaks|'
    r'psycho|lunatic|insane|'
    r'whore|slut|'
    r'n[i1]gg|'
    r'faggot|fag\b|'
    r'cunt|'
    r'pos\b|'
    r'rot in|burn in|go to hell|'
    r'subhuman|inferior|'
    r'vermin|pest|plague|cancer\b.*\bsociety|'
    r'deserve.*die|deserve.*suffer|'
    r'hope.*die|hope.*suffer|wish.*dead'
    r')\b',
    re.IGNORECASE
)

# High-warmth markers: care, encouragement, gratitude, love, support.
# Requiring 1+ match with warmth-specific phrases.
WARMTH_PATTERN = re.compile(
    r'\b('
    r'proud of you|so proud|'
    r'so happy for|happy for you|'
    r'grateful|thankful|thank you so much|'
    r'appreciate you|really appreciate|'
    r'wonderful person|wonderful thing|'
    r'amazing job|amazing person|you.re amazing|'
    r'beautiful soul|beautiful person|'
    r'love you|love that|i love|'
    r'care about you|care for you|i care|'
    r'support you|here for you|always here|'
    r'believe in you|i believe|'
    r'kind of you|so kind|very kind|'
    r'generous|generosity|'
    r'compassion|compassionate|'
    r'gentle|tenderness|tender|'
    r'warm|warmth|heartfelt|'
    r'congratulations|congrats|'
    r'well done|great job|good job|nice job|'
    r'keep going|keep it up|you got this|you can do|'
    r'never alone|not alone|'
    r'safe with|feel safe|'
    r'comfort|comforting|'
    r'blessing|blessed|'
    r'treasure|cherish|'
    r'adore|admire|'
    r'inspire|inspiring|inspired|'
    r'glad you|so glad|'
    r'sending love|sending hugs|'
    r'you deserve|you.re worth|'
    r'healing|heal from|'
    r'strength|so strong|you.re strong|'
    r'courage|courageous|brave'
    r')\b',
    re.IGNORECASE
)

# ESConv emotional support strategies that indicate high warmth
ESCONV_WARMTH_STRATEGIES = {
    'Affirmation and Reassurance',
    'Reflection of feelings',
    'Providing Suggestions',
    'Self-disclosure',
}


def load_existing_texts():
    """Load all texts already in psq.db to avoid duplicates."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    existing = set()
    for row in conn.execute('SELECT text FROM texts'):
        existing.add(row[0])
    conn.close()
    return existing


def score_hostility(text):
    """Count hostility keyword matches in text. Higher = more hostile."""
    return len(HOSTILITY_PATTERN.findall(text))


def score_warmth(text):
    """Count warmth keyword matches in text. Higher = warmer."""
    return len(WARMTH_PATTERN.findall(text))


def extract_esconv_supporter_text(raw_text):
    """Extract supporter utterances from ESConv raw JSON dialog.

    ESConv texts in the unlabeled pool are stored as raw JSON strings
    containing the full dialog. We extract only the supporter (sys)
    turns, which represent the emotionally supportive side.
    """
    try:
        dialog_data = json.loads(raw_text)
        dialog_turns = dialog_data.get('dialog', [])
        supporter_utterances = [
            turn['text']
            for turn in dialog_turns
            if turn.get('speaker') == 'sys'
        ]
        return ' '.join(supporter_utterances)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def extract_esconv_seeker_text(raw_text):
    """Extract seeker (help-seeking) utterances from ESConv dialog.

    Seeker utterances express emotional distress but in a
    help-seeking context — not hostile. Useful for mid-range HI.
    """
    try:
        dialog_data = json.loads(raw_text)
        dialog_turns = dialog_data.get('dialog', [])
        seeker_utterances = [
            turn['text']
            for turn in dialog_turns
            if turn.get('speaker') == 'usr'
        ]
        return ' '.join(seeker_utterances)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clean_empathetic_dialogues_text(text):
    """Clean EmpatheticDialogues formatting artifacts."""
    cleaned = text.replace('_comma_', ',')
    # Remove trailing metadata (e.g., ",5|5|5_5|5|5,\nhit:...")
    # Some ED texts have appended hit/conv metadata
    metadata_pattern = re.compile(r',\d\|\d\|\d_\d\|\d\|\d,\s*\nhit:.*$', re.DOTALL)
    cleaned = metadata_pattern.sub('', cleaned)
    return cleaned.strip()


def main():
    print("Loading existing texts from psq.db...")
    existing_texts = load_existing_texts()
    print(f"  {len(existing_texts)} texts already in DB")

    # ──────────────────────────────────────────────────────────────────────
    # Load unlabeled pool, separated by source, excluding existing texts
    # ──────────────────────────────────────────────────────────────────────
    print("\nLoading unlabeled pool...")
    pool_by_source = {}
    with open(UNLABELED_POOL_PATH) as file_handle:
        for line in file_handle:
            record = json.loads(line)
            text = record['text']
            source = record['source']
            if text not in existing_texts:
                if source not in pool_by_source:
                    pool_by_source[source] = []
                pool_by_source[source].append(text)

    for source, texts in sorted(pool_by_source.items()):
        print(f"  {source}: {len(texts)} available")

    # ──────────────────────────────────────────────────────────────────────
    # HIGH HOSTILITY: berkeley (Measuring Hate Speech dataset)
    # ──────────────────────────────────────────────────────────────────────
    print("\n--- Selecting high-hostility texts (HI score 0-2 candidates) ---")

    hostile_candidates = []
    for text in pool_by_source.get('berkeley', []):
        if len(text) < MINIMUM_TEXT_LENGTH:
            continue
        hostility_score = score_hostility(text)
        if hostility_score >= 2:
            hostile_candidates.append({
                'text': text,
                'source': 'berkeley',
                'keyword_matches': hostility_score,
            })

    # Sort by keyword density (matches per character) to prefer concentrated hostility
    hostile_candidates.sort(
        key=lambda candidate: candidate['keyword_matches'] / len(candidate['text']),
        reverse=True,
    )
    print(f"  Berkeley candidates with 2+ hostility matches: {len(hostile_candidates)}")

    # Select top candidates up to target
    selected_hostile = hostile_candidates[:TARGET_HIGH_HOSTILITY]
    print(f"  Selected: {len(selected_hostile)}")

    if selected_hostile:
        print(f"  Top match ({selected_hostile[0]['keyword_matches']} keywords): "
              f"{selected_hostile[0]['text'][:80]}...")
        print(f"  Lowest selected ({selected_hostile[-1]['keyword_matches']} keywords): "
              f"{selected_hostile[-1]['text'][:80]}...")

    # ──────────────────────────────────────────────────────────────────────
    # HIGH WARMTH: empathetic_dialogues + esconv
    # ──────────────────────────────────────────────────────────────────────
    print("\n--- Selecting high-warmth texts (HI score 8-10 candidates) ---")

    warm_candidates = []

    # EmpatheticDialogues: clean and score for warmth
    for text in pool_by_source.get('empathetic_dialogues', []):
        cleaned = clean_empathetic_dialogues_text(text)
        if len(cleaned) < MINIMUM_TEXT_LENGTH:
            continue
        warmth_score = score_warmth(cleaned)
        if warmth_score >= 1:
            warm_candidates.append({
                'text': cleaned,
                'source': 'empathetic_dialogues',
                'keyword_matches': warmth_score,
            })

    print(f"  EmpDial candidates with 1+ warmth matches: {len(warm_candidates)}")

    # ESConv: extract supporter utterances (emotionally supportive side)
    esconv_count = 0
    for text in pool_by_source.get('esconv', []):
        supporter_text = extract_esconv_supporter_text(text)
        if supporter_text and len(supporter_text) >= MINIMUM_TEXT_LENGTH:
            warmth_score = score_warmth(supporter_text)
            if warmth_score >= 1:
                warm_candidates.append({
                    'text': supporter_text,
                    'source': 'esconv',
                    'keyword_matches': warmth_score,
                })
                esconv_count += 1

    print(f"  ESConv supporter candidates with 1+ warmth matches: {esconv_count}")

    # Prosocial: supportive/encouraging responses
    prosocial_count = 0
    for text in pool_by_source.get('prosocial', []):
        if len(text) < MINIMUM_TEXT_LENGTH:
            continue
        warmth_score = score_warmth(text)
        if warmth_score >= 1:
            warm_candidates.append({
                'text': text,
                'source': 'prosocial',
                'keyword_matches': warmth_score,
            })
            prosocial_count += 1

    print(f"  Prosocial candidates with 1+ warmth matches: {prosocial_count}")

    # Sort by warmth keyword density
    warm_candidates.sort(
        key=lambda candidate: candidate['keyword_matches'] / len(candidate['text']),
        reverse=True,
    )
    print(f"  Total warm candidates: {len(warm_candidates)}")

    # Select top candidates up to target
    selected_warm = warm_candidates[:TARGET_HIGH_WARMTH]
    print(f"  Selected: {len(selected_warm)}")

    if selected_warm:
        source_counts = {}
        for candidate in selected_warm:
            source_counts[candidate['source']] = source_counts.get(candidate['source'], 0) + 1
        print(f"  Source breakdown: {source_counts}")
        print(f"  Top match ({selected_warm[0]['keyword_matches']} keywords): "
              f"{selected_warm[0]['text'][:80]}...")

    # ──────────────────────────────────────────────────────────────────────
    # Combine and write output
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n--- Writing output ---")

    # Deduplicate by text content (in case of overlap across sources)
    seen_texts = set()
    output_records = []

    for candidate in selected_hostile + selected_warm:
        text = candidate['text']
        if text not in seen_texts and text not in existing_texts:
            seen_texts.add(text)
            output_records.append({
                'text': text,
                'source': candidate['source'],
            })

    # Shuffle for labeling (avoids scorer bias from seeing all hostile then all warm)
    random.shuffle(output_records)

    with open(OUTPUT_PATH, 'w') as output_file:
        for record in output_records:
            output_file.write(json.dumps(record) + '\n')

    # Final summary
    source_summary = {}
    for record in output_records:
        source_summary[record['source']] = source_summary.get(record['source'], 0) + 1

    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Total records: {len(output_records)}")
    print(f"  Source breakdown: {source_summary}")
    print(f"  Purpose: HI range compression fix — extreme hostility + extreme warmth")
    print(f"  Next step: python scripts/label_separated.py extract --input {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

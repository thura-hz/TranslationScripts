#!/usr/bin/env python3
"""
movie_review_generator.py (Gemini version)

Generates a one-paragraph, warm-but-not-overly-casual Burmese movie review,
focused on plot with minimal spoilers — then critiques and revises its own
output multiple times to sharpen the final result.

Handles messy title input automatically:
    snake_case          -> "Snake Case"
    camelCase             -> "Camel Case"
    PascalCase            -> "Pascal Case"
    dot.separated.title   -> "Dot Separated Title"
    dash-separated-title  -> "Dash Separated Title"
    already Normal Title  -> left as-is

Release date is OPTIONAL. If you don't know it, just leave it blank —
the model will identify the movie and use its own knowledge of the release date.

Output file format: [review][MOVIE_NAME].txt, containing:

    [Translated by Tsuki Htet]

    [Movie Name] (released year)

    [Reviewed text]

Setup:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"   # get one at aistudio.google.com/apikey

Usage:
    python movie_review_generator.py "Inception" "2010"
    python movie_review_generator.py oppenheimer_2023
    python movie_review_generator.py spider.man.no.way.home
    python movie_review_generator.py theDarkKnight
    python movie_review_generator.py "Interstellar"          # no date, that's fine

    Or run with no arguments to be prompted interactively.
    Files are always saved (see format above). Use --no-save to skip saving.
    Add --rounds N to control how many critique/revise passes run (default 3).
    Add --verbose to print every draft, not just the final one.
"""

import os
import re
import sys
import argparse

try:
    from google import genai
except ImportError:
    sys.exit("Missing dependency. Run: pip install google-genai")

# `google.generativeai` (the old SDK) is fully deprecated — no updates, no bug
# fixes, and old model names like "gemini-2.5-flash" are being retired behind
# it. This script now uses the unified `google-genai` SDK instead.
MODEL_NAME = "gemini-3.5-flash"  # swap to "gemini-3.6-flash" for a cheaper/newer option, or "gemini-3.1-pro" for max quality
TRANSLATOR_NAME = "Tsuki Htet"  # change this if you want a different credit line

PROMPT_WITH_DATE = """Write a one-paragraph movie review in Burmese for {title} ({release_date}).
Write in a warm, natural, engaging tone — informative and inviting, like a
thoughtful short review someone would read before deciding to watch the film,
not slangy or overly casual, and not a dry formal critique either. Focus on
the movie's plot, mood, and what makes it worth watching, but avoid revealing
key twists, endings, or major spoilers — just enough to spark curiosity. Keep
the language flowing and idiomatic Burmese (not stiff or overly literal),
suitable for general audiences. Keep it to a single paragraph, in Burmese only,
with no English translation or extra commentary before or after."""

PROMPT_NO_DATE = """Write a one-paragraph movie review in Burmese for the movie "{title}".
I don't know the exact release date, so first identify the most likely/well-known
movie matching this title using your own knowledge. Write in a warm, natural,
engaging tone — informative and inviting, like a thoughtful short review someone
would read before deciding to watch the film, not slangy or overly casual, and
not a dry formal critique either. Focus on the movie's plot, mood, and what makes
it worth watching, but avoid revealing key twists, endings, or major spoilers —
just enough to spark curiosity. Keep the language flowing and idiomatic Burmese
(not stiff or overly literal), suitable for general audiences. Keep it to a
single paragraph, in Burmese only, with no English translation or extra
commentary before or after."""

CRITIQUE_PROMPT = """Here is a Burmese movie review draft for "{title}":

---
{draft}
---

Critique this draft strictly against these criteria:
1. Warm, natural, engaging tone — informative and inviting, but NOT slangy or
   overly casual (should read like a thoughtful short review, not a text
   message between close friends), and not dry/formal either
2. Focuses on plot and mood, giving a real sense of the story
3. No spoilers — no twists, no ending, no major plot-changing reveals
4. Idiomatic, naturally-written Burmese — not stiff, not a literal translation feel
5. Exactly one paragraph, Burmese only

List concrete weaknesses (if any), then rewrite the FULL review incorporating
the fixes. Output ONLY the revised paragraph in Burmese — no critique notes,
no headers, no English, no commentary before or after."""


def normalize_title(raw: str) -> str:
    """Convert snake_case, camelCase, PascalCase, dot.case, dash-case, etc. into a normal spaced title."""
    text = raw.strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"[._\-+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for word in text.split(" "):
        if word.isupper() or any(ch.isdigit() for ch in word):
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def extract_year(raw_title: str, raw_date):
    """Pull an embedded 4-digit year out of the title if no separate date was given."""
    if raw_date:
        return raw_title, raw_date
    match = re.search(r"(19|20)\d{2}", raw_title)
    if match:
        year = match.group(0)
        cleaned_title = (raw_title[: match.start()] + raw_title[match.end():]).strip("._- ")
        return cleaned_title, year
    return raw_title, raw_date


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY not set.\n"
            "Set it with: export GEMINI_API_KEY='your-key-here'\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def generate_draft(client, title: str, release_date) -> str:
    if release_date:
        prompt = PROMPT_WITH_DATE.format(title=title, release_date=release_date)
    else:
        prompt = PROMPT_NO_DATE.format(title=title)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text.strip()


def critique_and_revise(client, title: str, draft: str) -> str:
    prompt = CRITIQUE_PROMPT.format(title=title, draft=draft)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text.strip()


def refine(client, title: str, initial_draft: str, rounds: int, verbose: bool) -> str:
    draft = initial_draft
    if verbose:
        print(f"--- Draft 1 ---\n{draft}\n")
    for i in range(2, rounds + 2):  # rounds = number of critique/revise passes after the initial draft
        draft = critique_and_revise(client, title, draft)
        if verbose:
            print(f"--- Draft {i} ---\n{draft}\n")
    return draft


def build_output(title: str, release_date, review: str) -> str:
    header = f"[Translated by {TRANSLATOR_NAME}]"
    title_line = f"{title} ({release_date})" if release_date else title
    return f"{header}\n\n{title_line}\n\n{review}\n"


def make_filename(title: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in title).strip().replace(" ", "_")
    return f"[review][{safe_name}].txt"


def main():
    parser = argparse.ArgumentParser(description="Generate a Burmese movie review.")
    parser.add_argument("title", nargs="?", help="Movie title (any casing/format)")
    parser.add_argument("release_date", nargs="?", help="Release date (optional)")
    parser.add_argument("--no-save", action="store_true", help="Print only, don't save a file")
    parser.add_argument("--rounds", type=int, default=3, help="Number of critique/revise passes (default 3)")
    parser.add_argument("--verbose", action="store_true", help="Print every draft, not just the final one")
    args = parser.parse_args()

    raw_title = args.title or input("Movie title: ").strip()
    raw_date = args.release_date or input("Release date (leave blank if unknown): ").strip() or None

    raw_title, raw_date = extract_year(raw_title, raw_date)
    title = normalize_title(raw_title)

    print(f"\nTitle interpreted as: {title}" + (f" ({raw_date})" if raw_date else " (no date given)"))

    client = get_client()
    print(f"Generating and refining review ({args.rounds} revision passes)...\n")

    draft = generate_draft(client, title, raw_date)
    final_review = refine(client, title, draft, args.rounds, args.verbose)

    output_text = build_output(title, raw_date, final_review)

    if args.verbose:
        print("=== FINAL ===")
    print(output_text)

    if not args.no_save:
        filename = make_filename(title)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Saved to {filename}")


if __name__ == "__main__":
    main()